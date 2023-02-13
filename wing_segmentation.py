import os
import cv2 as cv
import numpy as np
import argparse

def extract_roi(path, padding=10, flip=False, output_folder=None):
    count=0
    img = cv.imread(path)
    basename = os.path.basename(path)[:-4]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.0001*dist_transform.max(),255,0)
    sure_fg= cv.morphologyEx(sure_fg,cv.MORPH_CLOSE,kernel,iterations=12)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers[unknown==255] = 0
    for label in range(1, ret):
        # Get the binary mask for each connected component
        mask = (markers == label).astype(np.uint8)
        # Get the contours for the binary mask
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the bounding rectangle for the contour
            x, y, w, h = cv.boundingRect(contour)
            # Check if the area of the bounding rectangle is larger than the threshold
            if w * h > img.shape[0] * img.shape[1] / 100:
                count+=1
                # Extract the ROI from the original image
                roi = img[(y-padding):(y+h+padding), (x-padding):(x+w+padding)]
                if x <= img.shape[1] / 2 and y <= img.shape[0] / 2:
                    name = 'upper_left_wing'
                elif x > img.shape[1] / 2 and y <= img.shape[0] / 2:
                    name = 'upper_right_wing'
                    if flip:
                        roi = cv.flip(roi, 1)
                        name += '_reflected'
                elif x <= img.shape[1] / 2 and y > img.shape[0] / 2:
                    name = 'lower_left_wing'
                elif x > img.shape[1] / 2 and y > img.shape[0] / 2:
                    name = 'lower_right_wing'
                    if flip:
                        roi = cv.flip(roi, 1)
                        name += '_reflected'
                img_file = f'{basename}_{name}_roi_{count}.jpg'
                cv.imwrite(os.path.join(output_folder, img_file), roi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ROIs from images in a folder")
    parser.add_argument('-i','--input', help='The path to the input folder')
    parser.add_argument('-o', '--output', help='The path to the output folder')
    parser.add_argument('--flip', help='Flip the right wings horizontally', action='store_true')
    parser.add_argument('-p','--padding', help='The padding around the ROI', default=10, type=int)
    args = parser.parse_args()
    
    # Create the output folder if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Iterate through the images in the input folder
    for image_filename in os.listdir(args.input):
        # Skip files that are not images
        ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        if not image_filename.lower().endswith(ext):
            continue

        input_path = os.path.join(args.input, image_filename)
        extract_roi(input_path, padding=args.padding, flip=args.flip, output_folder=args.output)