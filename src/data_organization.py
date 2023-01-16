import numpy as np
import pandas as pd
import os
from utils import read_config_file

config = read_config_file("config_data_org.yaml")

# read_path=config['image_path']

# #Read Meta Data and Filter CS
# df = pd.read_excel(filepath)
# cs = df[df["functional_area"]=="IT-Sware/DB/QA/Web/Graphics/GIS"].copy()

# # Read image from directory and save it in a list
# img_list = os.listdir(img_dir)
# job_code_list = [int(img.split(".")[0]) for img in img_list15]

# #find the intersect
# df_sample = df[df['job_code'].isin(job_code_list)].copy()

# #move to subfolder
# copy_to_cs2018_if_job_code_in_the_data_frame

# def main():