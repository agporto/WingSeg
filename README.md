# Wing Segmentation Tool

**This is not meant as a general tool. It is just a demonstration of image processing techniques for segmentation**

<p align="center"><img src="https://github.com/agporto/WingSeg/blob/master/images/demo.JPG" width="800"></p>


This CLI tool is designed to extract butterlfly wing ROIs from images in a folder and save them in another folder. The tool uses the extract_roi function to perform the image processing.

## Install 

- Clone the repo:
```
git clone https://github.com/agporto/WingSeg && cd WingSeg/
```

- Create a clean virtual environment:
```
conda create -n wingseg python=3.7
conda activate wingseg
```
- Install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
The tool can be used from the command line by running the following command:

```
python extract_roi_cli.py -i <input_folder> -o <output_folder> [--padding <padding>] [--flip]
```
Where:

- input_folder: The path to the folder containing the images to be processed.
- output_folder: The path to the folder where the extracted ROIs will be saved. If the folder does not exist, it will be created.
- padding (optional): The padding value to be used around the ROI when extracting it from the image. Default value is 10.
- flip (optional): A flag indicating if the ROIs from the right side of the animal should be flipped. Default value is False.

Output:

 - The tool will extract ROIs from the images in the input folder and save them as .jpg files in the output folder. The names of the saved files will include the name of the original image, the position of the ROI in the image (upper left wing, upper right wing, lower left wing, or lower right wing), and a unique count.

Example output file names:
```
image1_upper_left_wing_roi_1.jpg
image1_upper_right_wing_roi_2.jpg
image2_lower_left_wing_roi_1.jpg
image2_lower_right_wing_roi_2.jpg
```
