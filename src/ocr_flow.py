import numpy as np
import pandas as pd
import ocr_extraction
import ocr_evaluation
import preprocess_images

path = "E:/ADB_Project/code/data/cs_sample"

text = ocr_extraction.extract_text(path, n=10)
df = pd.DataFrame(text, index=np.arange(10))

i2t = list(df["ocrd_text"])

save_path = "E:/ADB_Project/code/data/pipeline_sample.csv"

df.to_csv(save_path, index=False)