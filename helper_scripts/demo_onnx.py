"""
Based on: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
"""
import numpy as np
import cv2
import argparse
import onnxruntime
from numpy import expand_dims
import matplotlib
from matplotlib import pyplot
from matplotlib.patches import Rectangle
matplotlib.use('TkAgg')


# User defined constants - todo: make into args
# define the anchors (here is tiny yolov4 anchors)
ANCHORS = [42,98, 53,120, 58,137, 76,181, 109,162, 131,261]

class BoundBox:
    """Class to keep track of bounding box object"""
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score
 
def _sigmoid(x):
    """Sigmoid function to get numbers into [0-1]"""
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    """Decode the candidate bounding boxes and class predictions"""
    netout = np.transpose(netout, (1, 2, 0))
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes
 
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """Correct the posistion of boxes and scale back to original image size"""
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
 
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
                return 0
        else:
            return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
    """Intersection over union (IOU) for two boxes"""
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    try:
        res = float(intersect) / union
    except ZeroDivisionError:
        res = 0
    return res
 
def do_nms(boxes, nms_thresh):
    """Non-maximum suppression (NMS) to elimate very closely overlapping boxes"""
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
 
def load_image_pixels(filename, input_shape):
    """Load and prepare an image for inferencing"""
    # load the image to get its shape
    image = cv2.imread(filename)
    height, width = image.shape[:2]
    image = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
    # bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # transpose for input shape to network
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # scale pixel values to [0, 1]
    image /= 255.0
    # add a dimension so that we have one sample (batch size)
    image = expand_dims(image, 0)
    return image, width, height
 
def get_boxes(boxes, labels, thresh):
    """Get all of the results above a threshold"""
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores
 
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    """Draw the results on original image and show"""
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()

def read_labels(labels_file):
    """Get the labels from names file"""
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    return [lab for lab in lines if len(lab) > 0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        help='Input image file')
    parser.add_argument('--thresh', type=float, default=0.7,
                        help='The confidence score threshold')
    parser.add_argument('--labels', type=str,
                        help='The labels file, e.g. obj.names')
    parser.add_argument('--model', type=str,
                        help='The ONNX model file')
    args = parser.parse_args()

    # load yolov4 model
    model = onnxruntime.InferenceSession(args.model)
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # load and prepare image
    image, image_w, image_h = load_image_pixels(args.image, (input_w, input_h))
    # make prediction
    # Compute
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: image})
    print('len of outputs = ', len(outputs))    # summarize the shape of the list of arrays
    print([a.shape for a in outputs])
    boxes = list()
    anchors = np.array(ANCHORS).reshape(2,-1)
    anchors = [anchors[1], anchors[0]]
    for i in range(len(outputs)):
        # decode the output of the network
        boxes += decode_netout(outputs[i][0], anchors[i], args.thresh, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = read_labels(args.labels)
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, args.thresh)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
    # draw what we found
    draw_boxes(args.image, v_boxes, v_labels, v_scores)