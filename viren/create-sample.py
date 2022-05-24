from os import listdir
from random import seed,sample
from zipfile import ZipFile

data_dirs = {'tour': 'data/cs_2020_filtered/', 'cs': 'data/cs_2020_filtered/'}

for cat in data_dirs:
    adverts_all = [advert for advert in listdir(data_dirs[cat])]
    seed(2022)
    adverts_sample = sample(adverts_all, 500)
    
    zip_file = ZipFile(cat + '_sample.zip', 'w')
    for advert in adverts_sample:
        zip_file.write(data_dirs[cat] + advert)
    zip_file.close()