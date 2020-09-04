#Script to call nii2png on a full directory of nii files and delete all but one image per scan.

import os
import keyboard
from os import listdir
from os.path import isfile, join
import subprocess

# Define input directories
AD_nii_directory = "C:\\Users\\dancu\\Desktop\\ADNI-Data\\ADNI\\nii\\AD\\"
CN_nii_directory = "C:\\Users\\dancu\\Desktop\\ADNI-Data\\ADNI\\nii\\CN\\"
# Define output paths
AD_png_directory = "C:\\Users\\dancu\\Desktop\\ADNI-Data\\ADNI\\png\\AD\\"
CN_png_directory = "C:\\Users\\dancu\\Desktop\\ADNI-Data\\ADNI\\png\\CN\\"
# Define lists containing only the images in said directories
AD_only_files = [f for f in listdir(AD_nii_directory) if isfile(join(AD_nii_directory, f))]
CN_only_files = [f for f in listdir(CN_nii_directory) if isfile(join(CN_nii_directory, f))]

# for file in CN_only_files:
	
# 	os.system("python nii2png.py -i " + CN_nii_directory + file + " -o " + CN_png_directory)
# 	CN_png_files = [f for f in listdir(CN_png_directory) if isfile(join(CN_png_directory, f))]

# 	for file in CN_png_files:

# 		if not file.endswith("z080.png"):
		
# 			os.remove(CN_png_directory + file)

for file in CN_only_files:
	
	os.system("python nii2png.py -i " + CN_nii_directory + file + " -o " + CN_png_directory)
	CN_png_files = [f for f in listdir(CN_png_directory) if isfile(join(CN_png_directory, f))]

	for file in CN_png_files:

		if not file.endswith("z076.png") and not file.endswith("z077.png") and not file.endswith("z078.png") and not file.endswith("z079.png") and not file.endswith("z080.png") and not file.endswith("z081.png") and not file.endswith("z082.png") and not file.endswith("z083.png"):
		
			os.remove(CN_png_directory + file)

