import os
from PIL import Image
import pandas as pd

basepath = "C:\\Code\\DeepFruit\\deepFruits_Datasets\\orange\\" 
datasetfile = "C:\\Code\\DeepFruit\\deepFruits_Datasets\\orange\\train_orange.txt"

f = open(datasetfile, "r") 
rows = []
for i, line in enumerate(f):
    line_splitted = line.split() 
    file_name = line_splitted[0]
    
    number_of_bboxes = int(line_splitted[1])
    bboxes = line_splitted[2:(number_of_bboxes*8)]
    im = Image.open(basepath+file_name)
    width, height = im.size    
    
    for xmin, ymin, xmax, ymax, label, score in zip(bboxes[0::6], bboxes[1::6], bboxes[2::6], bboxes[3::6], bboxes[4::6], bboxes[5::6]):
        annotation = []
        annotation.append(file_name.split("/")[1].replace(".png",".jpg")) # Filename
        annotation.append(width) # width
        annotation.append(height) # height
        annotation.append("orange") # label
        annotation.append(xmin) # xmin
        annotation.append(ymin) # ymin
        annotation.append(xmax) # xmax
        annotation.append(ymax) # ymax
        rows.append(annotation)

column_name = ['image_id', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
dataframe = pd.DataFrame(rows, columns=column_name)
dataframe.to_csv((basepath + '/train.csv'), index=None)
    
    
       