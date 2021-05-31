# Using the Darknet Framework with Azure ML

Note:  The purpose of this repo is to serve as a sample only and should not be considered production quality.  It may have future rapid iterations and additions of other algorithms/approaches for use with Azure ML.  Thank you for your patience.

The algorithm used here is [Darknet](https://github.com/AlexeyAB/darknet) Tiny YOLOv4.

## Prerequisites

1. Python 3.6+ installed locally
    - Recommend using a virtual environment or conda environment for this project to keep everything contained
2. [Azure ML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python)
3. Labeled data (YOLO format)
    - Convert to YOLO format as needed

## Instructions

1. Clone this repository.
2. Calculate anchor boxes.
    - You may find an anchor box calculator script [here](https://github.com/michhar/azure-and-ml-utils/blob/master/label_tools/calc_anchors_yolo_format.py).
3. Create `.azureml` folder in the root of this repository and download `config.json` from Azure ML Workspace resource in the Azure Portal to this folder.
4. Install the required packages.

```
pip install -r requirements_local.txt
```

5. Run Jupyter and then navigate to the given URL in a browser.

```
jupyter notebook
```

6. Open `Train.ipynb` and follow along (runs may be monitored from Azure ML Studio - also found at https://ml.azure.com).
    - Change the `num_classes` and `anchors` in the Train script writing cell.
    - Update the hyperparameter sweep values for your scenario to experiment.
    - Update the `epochs` for training to experiment.

## Todo

- [ ] Automatcially calcuate anchor boxes
- [ ] Support full sized YOLOv4
- [ ] Add hyperparameters and class number as params in `train.ipynb` Jupyter notebook
- [ ] Show converting to ONNX (for now please reference [this script](https://github.com/linghu8812/tensorrt_inference/blob/master/Yolov4/export_onnx.py))
