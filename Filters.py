
# USAGE:
# python Filters.py --input input/input.mp4 --output output/output.mp4 --filter sharp_back

# Import libraries
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import argparse
from moviepy.editor import *
from scipy.interpolate import UnivariateSpline


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video file")
ap.add_argument("-o", "--output", required=True, help="path to output video file")
ap.add_argument("-f", "--filter", required=True, choices=['grey_back', 'blur_back', 'bright_object', 'sharp_object',
                                                          'sharp_back', 'cartoon_back', 'warm_front_cold_black',
                                                          'edge_person', 'pencil_sketch_back'], help="filter to be used")
args = vars(ap.parse_args())


def brightnessControl(image, level):
    return cv2.convertScaleAbs(image, beta=level)


def auto_canny(image, sigma=0.33):
# compute the median of the single channel pixel intensities
    v = np.median(image)
# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
# return the edged image
    return cv2.Canny(image, lower, upper)


def cartoonizer(img_rgb):
    numDownSamples = 2       # number of downscaling steps
    numBilateralFilters = 7  # number of bilateral filtering steps
    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    # upsample image to original size
    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    # make sure resulting image has the same dims as original
    img_color = cv2.resize(img_color, (img_rgb.shape[1],img_rgb.shape[0]))

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)
    
# Warm/Cold
def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))
def warmImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))
def coldImage(image):
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))

#Pencil Sketch
def pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - gray_image
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
    def dodge(image, mask):
        return cv2.divide(image, 255-mask, scale=256)
    return dodge(gray_image,img_blur)

CLASS_NAMES = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# generate random (but visually distinct) colors for each class label
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)




class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # number of classes (we would normally add +1 for the background but the background class is *already* included in the class
    # names)
    NUM_CLASSES = len(CLASS_NAMES)



# initialize the inference configuration
config = SimpleConfig()



# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
model.load_weights('mask_rcnn_coco.h5', by_name=True)



# This function is used to change the colorful background information to grayscale.
# image[:,:,0] is the Blue channel,image[:,:,1] is the Green channel, image[:,:,2] is the Red channel
# mask == 0 means that this pixel is not belong to the object.
# np.where function means that if the pixel belong to background, change it to gray_image.
# Since the gray_image is 2D, for each pixel in background, we should set 3 channels to the same value to keep the grayscale.
def grey_back(image, mask):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:, :, 0] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 2]
    )
    return image


def blur_back(image, mask):
    blur_image = cv2.GaussianBlur(image, (35, 35), 0)
    blur_image[:,:][mask] = image[:,:][mask]
    return blur_image


def bright_object(image, mask):
    image[:,:][mask] = cv2.convertScaleAbs(image[:,:][mask], beta=100)
    return image


def sharp_object(image, mask):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image[:,:][mask] = cv2.filter2D(image[:,:][mask], -1, kernel)
    return image


def sharp_back(image, mask):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp_back = cv2.filter2D(image, -1, kernel)
    sharp_back[:,:][mask] = image[:,:][mask]
    return sharp_back


def cartoon_back(image, mask):
    back = cartoonizer(image)
    back[:,:][mask] = image[:,:][mask]
    return back

def warm_front_cold_black(image, mask):
    back = coldImage(image)
    front = warmImage(image)
    back[:,:][mask] = front[:,:][mask]
    return back

def edge_person(image, mask):
    copy = image.copy()
    edge = auto_canny(image)
    edge = cv2.dilate(edge, None, iterations=2)
    cnts, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea)
    c = c[-2:]
    cv2.drawContours(copy, c, -1, (126,249,255), 3)
    image[:,:][mask] = copy[:,:][mask]
    return image

def pencil_sketch_back(image, mask):
    back = pencil_sketch(image)
    image[:, :, 0] = np.where(
        mask == 0,
        back[:, :],
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        back[:, :],
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        back[:, :],
        image[:, :, 2]
    )
    return image
    
def make_frame(frame, ftr):
    # perform a forward pass of the network to obtain the results
    r = model.detect([frame], verbose=1)[0]
    # loop over of the detected object's bounding boxes and masks
    max_area = 0
    for i in range(len(r['class_ids'])):
        # extract only the largest 'person' and mask for the current detection, then grab the color to visualize 
        # the mask (in BGR format)
        if(CLASS_NAMES[r['class_ids'][i]]=='person'):
            (startY, startX, endY, endX) = r["rois"][i]
            area = (startY-endY)*(startX-endX)
            if(area>max_area):
                max_area = area
                mask = r["masks"][:, :, i]
            else:
                continue
        else:
            continue
    return ftr(frame.copy(), mask)


def my_transformation(clip, ftr):
    def new_transformation(frame):
        return make_frame(frame, ftr)
    return clip.fl_image(new_transformation)


clip = VideoFileClip(args['input'])
if(args['filter']=='grey_back'):
    modifiedClip = clip.fx(my_transformation, grey_back)
if(args['filter']=='blur_back'):
    modifiedClip = clip.fx(my_transformation, blur_back)
if(args['filter']=='bright_object'):
    modifiedClip = clip.fx(my_transformation, bright_object)
if(args['filter']=='sharp_object'):
    modifiedClip = clip.fx(my_transformation, sharp_object)
if(args['filter']=='sharp_back'):
    modifiedClip = clip.fx(my_transformation, sharp_back)
if(args['filter']=='cartoon_back'):
    modifiedClip = clip.fx(my_transformation, cartoon_back)
if(args['filter']=='warm_front_cold_black'):
    modifiedClip = clip.fx(my_transformation, warm_front_cold_black)
if(args['filter']=='edge_person'):
    modifiedClip = clip.fx(my_transformation, edge_person)
if(args['filter']=='pencil_sketch_back'):
    modifiedClip = clip.fx(my_transformation, pencil_sketch_back)


modifiedClip.write_videofile(args['output'])

