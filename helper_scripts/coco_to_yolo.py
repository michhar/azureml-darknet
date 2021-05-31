"""
Convert COCO format and image folder to YOLO format and
folder structure.

Caution: it will overwrite the contents of previous conversions if
using the same output project name.

For a list of args and help:

python3 coco_to_yolo.py --help

Details of bounding box conversion (both with values between 0-1):
COCO bounding box: (x-top left, y-top left, width, height)
YOLO bounding box: [x-center, y-center, width, height]

Note:  bounding boxes in both cases are measured from
top, left of image.
"""
import json
import argparse
import os
from PIL import Image
import glob
import random
import shutil


def check_images(image_dir, outdir):
    """Get a list of true images in folder and copy those images over 
    to the output annot folder"""
    dirfiles = glob.glob(os.path.join(image_dir, '*.*'))
    trueimages = []
    for imagef in dirfiles:
        if is_image(imagef):
            imagef_short = imagef.split(os.sep)[-1]
            trueimages.append(imagef_short)
            # Copy image file to annot folder
            shutil.copy(imagef, os.path.join(outdir, 'img'))
    print('Looks like you have {} image files.'.format(len(trueimages)))
    return trueimages

def is_image(imgfile):
    """Checks to see if file is an image and returns True or False"""
    try:
        img = Image.open(imgfile)
        if img is not None:
            isimage = True
        else:
            print('Found a non-image in image folder.')
            isimage = False
    except Exception:
        print('Found a non-image in image folder.')
        isimage = False
    return isimage

def make_dirs(outdir):
    """Make the directories for the YOLO format export"""
    # Create dirs if they do not exist
    try:
        if ' ' in outdir:
            assert False
    except Exception:
        print('Please use an output directory name that does not contain spaces.')

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=False)
    
    imagedir = os.path.join(outdir, 'img')
    if not os.path.exists(imagedir):
        os.makedirs(imagedir, exist_ok=False)    

def read_coco(cocofile):
    """Read coco annotation file (json format) into dictionaries"""
    # Read file
    with open(cocofile, 'r') as fptr:
        cocojson = json.load(fptr)
    
    imgid_to_annot = {}
    # Create dict of image id to filename and annots for all images
    imgid_to_filename = {img['id']: (img['file_name'], img['width'], 
                             img['height']) for img in cocojson['images']}
    catid_to_category = {(cat['id']-1): cat['name'] for cat in 
                             cocojson['categories']}
    for annot in cocojson['annotations']:
        image_id = annot['image_id']
        if image_id in imgid_to_annot:
            info = imgid_to_annot[image_id]
            info.append((annot['bbox'], annot['category_id']-1))
            imgid_to_annot[image_id] = info
        else:
            imgid_to_annot[image_id] = [(annot['bbox'], 
                                         annot['category_id']-1)]

    return imgid_to_filename, imgid_to_annot, catid_to_category

def output_annots_to_yolo(imgid_to_filename,
                          imgid_to_annot,
                          catid_to_category,
                          outdir,
                          trueimages):
    """Output to YOLO format"""
    listofimages = []
    for imageid in imgid_to_annot:
        # Image file info
        imagefile, _, _ = imgid_to_filename[imageid]
        # Use unix file paths because of Azure ML (unix-based)
        imagefile = imagefile.split('/')[-1]
        if not imagefile in trueimages:
            continue
        imagefilelong =  'data/img/' + imagefile
        # For train.txt and valid.txt
        listofimages.append(imagefilelong)
        random.shuffle(listofimages)

        # Annotation info and print one txt file per image
        annots_for_image = imgid_to_annot[imageid]
        txt_file = '.'.join(imagefile.split('.')[:-1]) + '.txt'
        with open(os.path.join(outdir, 'img', txt_file), 'w') as fout:
            for annot in annots_for_image:
                bbox, categoryid = annot
                xc, yc, w, h = x1y1wh_to_xcycwh(bbox)
                annotline = ' '.join([str(categoryid), str(xc), str(yc), 
                                          str(w), str(h)]) + '\n'
                fout.write(annotline)

        # Print train.txt and valid.txt
        # Hardcoded: take 90% images for train set
        lentrainset = int(len(listofimages)*0.9)
        trainset = listofimages[:lentrainset]
        valset = listofimages[lentrainset:]
        for txt_file in ['train.txt', 'valid.txt']:
            with open(os.path.join(outdir, txt_file), 'w') as fout:
                if txt_file == 'train.txt':
                    settoprint = trainset
                else:
                    settoprint = valset
                for img in settoprint:
                    fout.write(img + '\n')

    # Category name files
    with open(os.path.join(outdir, 'obj.names'), 'w') as fout:
        for i in range(len(catid_to_category)):
            fout.write(catid_to_category[i] + '\n')
    
    # Write the data info file
    with open(os.path.join(outdir, 'obj.data'), 'w') as fout:
        fout.write("""classes = {}
train = data/train.txt
valid = data/valid.txt
names = data/obj.names
backup = outputs/
        """.format(len(catid_to_category.keys())))

def x1y1wh_to_xcycwh(bbox):
    """Convert bounding boxes from
    [x_top, y_top, width, height] to
    [x_center, y_center, width, height]
    
    Note: coordinates measured from top,
    left corner of image and are normalized
    0-1 which we will keep"""
    x_top, y_top, w_box, h_box = bbox
    x_center = x_top + (0.5*w_box)
    y_center = y_top + (0.5*h_box)

    return x_center, y_center, w_box, h_box


def main():
    """Driver function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-file', dest='cocofile', type=str,
                       help='The single coco format annotation file')
    parser.add_argument('--image-dir', dest='imagedir', type=str,
                       help='The single directory containing the image files')                      
    parser.add_argument('--project-name', dest='projname', type=str, default='data_project',
                       help='The project name - no spaces in name please; will overwrite')
    args = parser.parse_args()

    outdir = 'data_'+args.projname
    make_dirs(outdir)
    listtrueimages = check_images(args.imagedir, outdir)
    imgid_to_filename, imgid_to_annot, catid_to_category = read_coco(args.cocofile)

    # Loop over images, find all annots, output to txt
    output_annots_to_yolo(imgid_to_filename,
                          imgid_to_annot,
                          catid_to_category,
                          outdir,
                          listtrueimages)

if __name__ == '__main__':
    main()