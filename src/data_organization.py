import numpy as np
import pandas as pd
import os
from utils import read_config_file
import shutil
import argparse

def run(config_file_path):
    "Runs the data subsetting operation"

    config = read_config_file(config_file_path)

    image_path = config['image_path']
    data_path = config['data_path']
    years = config['list_of_years']
    sectors = config['list_of_sectors']
    sec_shorts = config['sector_acronyms']

    # Read the meta data file
    df = pd.read_excel(data_path)

    for year in years:
        for index, sector in enumerate(sectors):
            # Filter the data frame by the desired sectors as given in the config file
            df_sector = df[df["functional_area"]==sector].copy()

            # Get a full list of images from the folder containing the relevant year
            full_img_list = os.listdir(os.path.join(image_path, year))

            # Get the job code list for matching purposes from the image list
            job_code_list = [int(img.split(".")[0]) for img in full_img_list]

            # Filter the already filtered by data frame (by sector) by the relevant year
            df_sector_year = df_sector[df_sector['job_code'].isin(job_code_list)].copy()

            # create array to conduct the intersect1D operation with numpy
            #img_list_array = np.array(full_img_list)
            job_code_array = np.array(job_code_list)

            # get the intersection and save the output in a list of images for the relevant sector and the year
            intersect_list = list(np.intersect1d(job_code_array, np.array(df_sector_year['job_code'])))
            sector_year_img_list = [img for img in full_img_list if int(img.split(".")[0]) in intersect_list]

            # create the relevant sectoral folder within the year folder
            new_directory = f"{os.path.join(image_path, year)}/{sec_shorts[index]}"
            os.mkdir(new_directory)

            # conduct the copy operation
            for img in sector_year_img_list:
                src_file = os.path.join(image_path, year, img)
                dst_file = os.path.join(image_path, year, sec_shorts[index], img)
                # copy + Paste to the relevant folder
                shutil.copy(src_file, dst_file)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--configfile",
        type=str,
        help="Path to the configuration file (required)",
        required=True
    )

    args = arg_parser.parse_args()

    run(args.configfile)