import os
from PIL import Image

# Define the source folder path
source_folder = ''

# Iterate through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.tif'):
        # Construct the full file path
        tif_file_path = os.path.join(source_folder, filename)
        # Open and read the tif file
        with Image.open(tif_file_path) as img:
            # Construct the new jpg file path
            jpg_filename = filename.replace('.tif', '.jpg')
            jpg_file_path = os.path.join(source_folder, jpg_filename)

            # Save the image in jpg format
            img.save(jpg_file_path, 'JPEG')

print("Conversion completed!")
