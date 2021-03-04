"""
Azure ML train script for Darknet object detection experiment
"""
import os
import requests
import subprocess
import shutil
import argparse

from azureml.core.run import Run
from azureml.core.model import Model


# Azure ML run to log metrics etc.
run = Run.get_context()

# Fill in number of classes
num_classes = 1

# Fill in with a list of anchor boxes
anchors = "11, 33,  18, 28,  19, 33,  22, 30,  22, 34,  25, 34"

# Test greeting
def greeting():
    print("Welcome to darknet container!")
greeting()

os.makedirs('./outputs', exist_ok=True)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str,
                    dest='data_folder', help='data folder')
parser.add_argument('--lr', type=float, default=0.001,
                    dest='lr', help='learning rate')
parser.add_argument('--bs', type=int, default=4,
                    dest='bs', help='minibatch size')
parser.add_argument('--epochs', type=int, default=4,
                    dest='epochs', help='number of epochs')
args = parser.parse_args()

# ========================== Get data ==========================

# Look at data folder
print('Data folder is at:', args.data_folder)
print('List all files: ', os.listdir(args.data_folder))

zipfilename = os.path.join(args.data_folder, "data.zip")
# Unzip the data.zip to cwd where it will be a folder called "data"
shutil.unpack_archive(zipfilename, ".")
fulldatapath = os.path.join(args.data_folder, "data")

# ========================== Create or download necessary files ==========================

# Make obj.data file (if we put model in "outputs" it will show in the Portal)
obj_data = """
classes= {}
train  = data/train.txt
valid  = data/valid.txt
names = data/obj.names
backup = outputs/
""".format(num_classes)

print("Making obj.data...") # see top of notebook
with open("obj.data", "w") as f:
    f.write(obj_data)
    
# Get the pre-trained weights file
print("Getting the pre-trained weights file, yolov4-tiny.conv.29...")
url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
response = requests.get(url)
if response.status_code == 200:
    with open("yolov4-tiny.conv.29", "wb") as f:
        f.write(response.content)

print("Creating the config file...")
# Get the config file
url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
response = requests.get(url)
if response.status_code == 200:
    with open("yolov4-tiny.cfg", "wb") as f:
        f.write(response.content)
with open("yolov4-tiny.cfg", "r") as f:
    config_content = f.read()

# Replace LR
config_content = config_content.replace("learning_rate=0.00261", "learning_rate={}".format(args.lr))
# Replace number of filters in CN layer before yolo layer
num_filters = (num_classes+5)*3
config_content = config_content.replace("filters=255", "filters={}".format(num_filters))
# Replace number of classes
config_content = config_content.replace("classes=80", "classes={}".format(num_classes))
# Replace batch size
config_content = config_content.replace("batch=64", "batch={}".format(args.bs))
# Replace max batches/epochs and learning rate stepping epochs
config_content = config_content.replace("max_batches = 2000200", "max_batches={}".format(args.epochs))
config_content = config_content.replace("steps=1600000,1800000", 
                                        "steps={},{}".format(int(0.8*args.epochs), 
                                                             int(0.9*args.epochs)))
# Replace anchors
config_content = config_content.replace("10,14,  23,27,  37,58,  81,82,  135,169,  344,319",
                                       anchors)

with open("yolov4-tiny-custom.cfg", "w") as f:
    f.write(config_content)
# What does the config file look like now?
os.system("cat yolov4-tiny-custom.cfg")

# What is our current working directory?
print("Current working directory: {}".format(os.getcwd()))
print("Contents of directory: ")
os.system("ls")

# ========================== Train model ==========================

# Predict with darknet
print("Running darknet training experiment for {} epochs!".format(args.epochs))
# os.system("darknet detector train obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show -clear")

result = subprocess.run(['darknet', 
                         'detector',
                         'train',
                         'obj.data',
                         'yolov4-tiny-custom.cfg', 
                         'yolov4-tiny.conv.29',
                         '-map',
                         '-dont_show',
                         '-clear'], 
                        stdout=subprocess.PIPE).stdout.decode('utf-8')

os.system("ls")

mAP = ""
# # Capture mAP of final model (read from log file)
# with open("./azureml-logs/70_driver_log.txt", "r") as f:
#     logfile_content = f.readlines()
result = result.split("\n")
for line in result:
    # This is what darknet outputs at end
    if "mean average precision (mAP@0.50)" in line:
        mAP = float(line[-8:].replace(" % ", ""))
        # Log to Azure ML workspace
        run.log('mAP0.5_all', mAP)

        
# ========================== Register model - TBD ==========================

# # Get class names as string
# with open("./data/obj.names", "r") as f:
#     class_names = f.read().replace("\n", "_").replace(" ", "").strip()

# model = run.register_model(model_name='darknet-yolov4-tiny',
#                            tags={"mAP0.5_all": mAP,
#                                  "classes": class_names,
#                                  "learning_rate": args.lr,
#                                  "batch_size": args.bs,
#                                  "format": "darknet"},
#                            model_path="./outputs/yolov4-tiny-custom_final.weights")
                           
# ========================== Evaluate model - TBD ==========================



# ========================== Small test - optional ==========================

with open("data/valid.txt", "r") as f:
    validtxt = f.readlines()
# Pick first image
testimg = validtxt[0].strip()

# Predict with darknet
print("Running darknet detector test!")
os.system("darknet detector test obj.data yolov4-tiny-custom.cfg ./outputs/yolov4-tiny-custom_final.weights -thresh 0.25 {} -ext_output".format(testimg))
os.system("ls")

if os.path.exists("predictions.jpg"):
    shutil.copyfile("predictions.jpg", "./outputs/predictions.jpg")
if os.path.exists("predictions.png"):
    shutil.copyfile("predictions.png", "./outputs/predictions.png")

# ========================== Convert model to tflite ==========================

# Set up project
setup_tflite = """
    git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
"""

os.system(setup_tflite)

with open("tensorflow-yolov4-tflite/core/config.py", "r") as f:
    config_tflite = f.read()
config_tflite = config_tflite.replace("coco.names", "obj.names")
config_tflite = config_tflite.replace("__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]",
                                      "__C.YOLO.ANCHORS_TINY         = [{}]".format(anchors))
with open("tensorflow-yolov4-tflite/core/config.py", "w") as f:
    f.write(config_tflite)
shutil.copy("data/obj.names", "tensorflow-yolov4-tflite/data/classes/obj.names")

# Save as TF
os.system("cd tensorflow-yolov4-tflite && python save_model.py --weights ../outputs/yolov4-tiny-custom_best.weights --output ../outputs/yolov4-tiny-416-tflite --input_size 416 --model yolov4 --framework tflite --tiny")
# Convert to TFLite
os.system("cd tensorflow-yolov4-tflite && python convert_tflite.py --weights ../outputs/yolov4-tiny-416-tflite --output ../outputs/yolov4-tiny-416-fp16.tflite --quantize_mode float16")

# model = run.register_model(model_name='tflite-yolov4-tiny',
#                            tags={"classes": class_names,
#                                  "format": "tflite"},
#                            model_path=os.path.join("./outputs/yolov4-tiny-416-fp16.tflite"))
