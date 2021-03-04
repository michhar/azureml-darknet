"""
Azure ML train script for Darknet object detection experiment
"""
import os
import requests
import subprocess
import shutil
import argparse


# Fill in number of classes
num_classes = 1

# Create a list with anchor boxes
anchors = []

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
os.system("darknet detector train obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show -clear")
os.system("ls")

# ========================== Evaluate model - TBD ==========================



# ========================== Small test - optional ==========================

# print("Getting the test image...")
# # Get a test image
# url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/giraffe.jpg"
# response = requests.get(url)
# if response.status_code == 200:
#     with open("giraffe.jpg", "wb") as f:
#         f.write(response.content)

# # Predict with darknet
# print("Running darknet detector test!")
# os.system("darknet detector test coco.data yolov4.cfg yolov4.weights -thresh 0.25 giraffe.jpg -ext_output")
# os.system("ls")

# if os.path.exists("predictions.jpg"):
#     shutil.copyfile("predictions.jpg", "outputs/predictions.jpg")
# if os.path.exists("predictions.png"):
#     shutil.copyfile("predictions.png", "outputs/predictions.png")

# ========================== Convert model to tflite - TBD ==========================

