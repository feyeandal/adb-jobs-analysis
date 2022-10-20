import pandas as pd
import numpy as np
from PIL import Image, ImageShow, ImageDraw
from os import listdir

all_adverts = [advert for advert in listdir('data/tour_sample')]
rng = np.random.default_rng(2022)
rng.shuffle(all_adverts)

pd.DataFrame(data={'advert': all_adverts}) \
    .to_csv('data/tour_tags.csv', index=True)

i = 0
ImageShow.register(ImageShow.WindowsViewer(), 0)

for file_name in reversed(all_adverts[i:i+10]):
    advert = Image.open('data/tour_sample/' + file_name)
    ImageDraw.Draw(advert).text(
        xy=(15, 15),
        text=file_name,
        fill='black',
        anchor='mm')
    advert.show(title=file_name)
i = i + 10