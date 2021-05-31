"""
Calculate anchor boxes for YOLO blocks - use with CAUTION as
the anchor box sizes could greatly affect the training results.

The input format is YOLO (commonly used with Darknet).
"""
import os
import numpy as np
from PIL import Image
import glob
import argparse


class YOLO_Kmeans:
    """A class to calculate anchor box sizes"""

    def __init__(self, cluster_number, out_file, img_dir, resolution):
        self.cluster_number = int(cluster_number)
        self.out_file = out_file
        self.img_dir = img_dir
        self.resolution = resolution

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        try:
            clusters = boxes[np.random.choice(
                box_number, k, replace=False)]  # init k clusters
        except ValueError as err:
            print('Kmeans error: ', err)
            assert False
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        """Write KMeans anchor results to output file"""
        f = open(self.out_file, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        """
        Get the width and height of an image by name.
        Get the annotation by image name.
        Returns two dictionaries.
        """
        imagename_to_wh = {}
        labelname_to_annot = {}
        data_set = []
        # img dir has the images and labels
        filelist = glob.glob(self.img_dir + os.sep + "*.*")
        # Get only image files, filter out .txt files
        images = [x for x in filelist if x.split('.')[-1] != "txt"]
        #annots = [x for x in filelist if x.split('.')[-1] == "txt"]
        # for image in images:
        #     # Dict key
        #     imagename = os.path.basename(image)
        #     img = Image.open(image)
        #     imagename_to_wh[imagename] = img.size
        for annot in images:
            # dict key
            imagename = os.path.basename(annot)
            # Generate the annot name from the image name as they are same except suffix
            annot = ".".join(annot.split(".")[:-1]) + ".txt"
            with open(annot, "r") as f:
                annotlines = f.readlines()
                for line in annotlines:
                    label, x_center, y_center, box_width, box_height = line.split(" ")
                    box_width, box_height = float(box_width), float(box_height)
                    box_width *= self.resolution #imagename_to_wh[imagename][0]
                    box_height *= self.resolution #imagename_to_wh[imagename][1]
                    box_width, box_height = int(box_width), int(box_height)
                    data_set.append([box_width, box_height])
                    print([box_width, box_height])
        return np.array(data_set)

    def txt2clusters(self):
        """Driver method"""
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":

    # For command line options
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--img-dir', type=str, dest='img_dir', default='data/img',
        help="The directory of image files and .txt files in YOLO format"
    )
    parser.add_argument(
        '--out-file', type=str, dest='out_file', default='anchors.txt',
        help="The output file containing the anchor boxes"
    )
    parser.add_argument(
        '--anchor-num', type=int, dest='anchor_num', default=6,
        help="The number of anchors (either 6 or 9)"
    )
    parser.add_argument(
        '--network-size', type=int, dest='size', default=416,
        help="The size of the input into the network (usually 416 or 512).\
            The size must represent height and width so must be square."
    )

    args = parser.parse_args()

    kmeans = YOLO_Kmeans(cluster_number=args.anchor_num,
                         out_file=args.out_file,
                         img_dir=args.img_dir,
                         resolution=args.size)
    kmeans.txt2clusters()