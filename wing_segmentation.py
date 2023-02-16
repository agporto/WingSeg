import os
import cv2 as cv
import numpy as np
import argparse


def extract_roi(path, padding=10, flip=False, output_folder=None, skip_if_exists = False):
    count = 0
    img = cv.imread(path)
    height, width, _ = img.shape

    if height > width:
        print(f"Image {path} is rotated. It will be ignored.")
        return

    basename = os.path.basename(path)[:-4]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
    sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_CLOSE, kernel, iterations=10)

    # Finding connected components
    sure_fg = np.uint8(sure_fg)
    output = cv.connectedComponentsWithStats(sure_fg, 4, cv.CV_32S)
    (numMarkers, markers, stats, centroids) = output

    # Get the total area of the image
    total_area = img.shape[0] * img.shape[1]

    # Filter out markers that are too small
    labels = 0
    clean_stats = []
    for i in range(1, numMarkers):
        area = stats[i, cv.CC_STAT_AREA]
        shape = stats[i, cv.CC_STAT_WIDTH] / stats[i, cv.CC_STAT_HEIGHT]
        bbox_area = stats[i, cv.CC_STAT_WIDTH] * stats[i, cv.CC_STAT_HEIGHT]
        if area > (total_area / 50) and area > (bbox_area / 4) and shape > 0.33 and shape < 3:
            labels += 1
            markers[markers == i] = labels


            clean_stats.append(stats[i])
        else:
            markers[markers == i] = 0

    # Iterate through the markers and extract the ROI
    for label in range(0, labels):
        count += 1
        x, y, w, h, area = clean_stats[label]
        #Check if x or y negative
        x1 = max (0, x - padding)
        y1 = max (0, y - padding)
        x2 = min (x + w + padding, img.shape[1])
        y2 = min (y + h + padding, img.shape[0])
        #find center of the rectangle
        xc = x + w / 2
        yc = y + h / 2

        # Extract the ROI from the original image
        

        if xc <= img.shape[1] / 2 and yc <= img.shape[0] / 2:
            roi = img[y1:y2,x1:x2]
            name = "upper_left_wing"

        elif xc > img.shape[1] / 2 and yc <= img.shape[0] / 2:
            name = "upper_right_wing"
            roi = img[y1:y2,x1:x2]
            if flip:
                roi = cv.flip(roi, 1)
                name += "_reflected"

        elif xc <= img.shape[1] / 2 and yc > img.shape[0] / 2:
            name = "lower_left_wing"
            y1 = max (0, y1 - padding)
            roi = img[(y1 - padding):y2,x1:x2]

        elif xc > img.shape[1] / 2 and yc > img.shape[0] / 2:
            name = "lower_right_wing"
            y1 = max (0, y1 - padding)
            roi = img[(y1 - padding):y2,x1:x2]
            if flip:
                roi = cv.flip(roi, 1)
                name += "_reflected"

        img_file = f"{basename}_{name}_roi_{count}.jpg"
        out_path = os.path.join(output_folder, img_file)
        #check if path exists
        if os.path.exists(out_path) and skip_if_exists:
            print(f"File {out_path} already exists. Skipping.")
        else:
            cv.imwrite(os.path.join(output_folder, img_file), roi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ROIs from images in a folder")
    parser.add_argument("-i", "--input", help="The path to the input folder")
    parser.add_argument("-o", "--output", help="The path to the output folder")
    parser.add_argument(
        "--flip", help="Flip the right wings horizontally", action="store_true"
    )
    parser.add_argument(
        "-s","--skip", help="Skip writing the file to disk if it exists", action="store_true"
    )
    parser.add_argument(
        "-p", "--padding", help="The padding around the ROI", default=75, type=int
    )
    args = parser.parse_args()

    # Create the output folder if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Iterate through the images in the input folder
    for image_filename in os.listdir(args.input):

        # Skip files that are not images
        ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        if not image_filename.lower().endswith(ext):
            continue

        input_path = os.path.join(args.input, image_filename)
        extract_roi(
            input_path, padding=args.padding, flip=args.flip, output_folder=args.output, skip_if_exists=args.skip
        )
