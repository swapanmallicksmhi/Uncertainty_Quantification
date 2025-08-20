# This script will create new file after adjustment and crop.
# Define the cropping values Pixels to crop
# Swapan Mallick on 11 June 2025
#

import os
import argparse
import imageio
import matplotlib.pyplot as plt
from PIL import Image

def process_data(input_folder, output_folder):
    # Define the cropping values
    #left_crop = 180; right_crop = 240
    #top_crop = 80; bottom_crop = 140
    #-------------for q all----
    left_crop = 210; right_crop = 215
    top_crop = 80; bottom_crop = 140

    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            img = imageio.imread(input_path)
            height, width, channels = img.shape

            # Crop the image
            cropped_img = img[top_crop:height - bottom_crop,
                              left_crop:width - right_crop, :]

            # Convert to PIL for saving
            pil_image = Image.fromarray(cropped_img)

            # Save image
            plt.imshow(pil_image)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight',
                        facecolor='white', dpi=100)
            plt.close()

    print(f"Processing complete. Cropped images are saved in '{output_folder}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and adjust PNG/JPG images.")
    parser.add_argument("--input", required=True, help="Path to input folder containing images")
    parser.add_argument("--output", required=True, help="Path to save cropped images")

    args = parser.parse_args()
    process_data(args.input, args.output)
