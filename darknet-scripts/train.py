"""
Azure ML training script for Darknet object detection experiment
"""
import os
import requests
import subprocess
import shutil


# Test container
def greeting():
    print("Welcome to darknet container!")

greeting()

print("Getting the test image...")
# Get a test image
url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/giraffe.jpg"
response = requests.get(url)
if response.status_code == 200:
    with open("giraffe.jpg", "wb") as f:
        f.write(response.content)

# Make coco.data file
coco_data = """
classes= 80
train  = train.txt
valid  = val.txt
names = coco.names
backup = backup/
eval=coco
"""
print("Making coco.data")
with open("coco.data", "w") as f:
    f.write(coco_data)

print("Getting coco.names...")
# Get the names file
url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"
response = requests.get(url)
if response.status_code == 200:
    with open("coco.names", "wb") as f:
        f.write(response.content)

print("Getting the weights file, yolov4.weights...")
# Get the weights file
url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
response = requests.get(url)
if response.status_code == 200:
    with open("yolov4.weights", "wb") as f:
        f.write(response.content)

print("Getting the config file...")
# Get the config file
url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
response = requests.get(url)
if response.status_code == 200:
    with open("yolov4.cfg", "wb") as f:
        f.write(response.content)

# What is our current working directory?
print("Current working directory: {}".format(os.getcwd()))
print("Contents of directory: ")
os.system("ls")

# Predict with darknet
print("Running darknet detector test!")
os.system("darknet detector test coco.data yolov4.cfg yolov4.weights -thresh 0.25 giraffe.jpg -ext_output")
os.system("ls")

if os.path.exists("predictions.jpg"):
    shutil.copyfile("predictions.jpg", "outputs/predictions.jpg")
if os.path.exists("predictions.png"):
    shutil.copyfile("predictions.png", "outputs/predictions.png")


print("Return value: {}".format(retval))
