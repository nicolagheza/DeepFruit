from sklearn.utils import shuffle

import os
from PIL import Image
import pandas as pd

basepath = "C:\\Code\\DeepFruit\\datasets" 
train_files = []
labels = []
for fruit in os.listdir(basepath):
        labels.append(fruit)
        train_file = "{}\\{}\\test_RGB.txt".format(basepath, fruit)
        train_files.append(train_file)

# Create CSV annotations for Faster-RCNN with Luminoth
def createCSV(train_files):
        rows = []
        for n, file in enumerate(train_files):
                for i, line in enumerate(open(file)):
                        line_splitted = line.split() 
                        file_name = line_splitted[0]
                        label = labels[n]
                        number_of_bboxes = int(line_splitted[1])
                        bboxes = line_splitted[2:(number_of_bboxes*8)]
                        im = Image.open("{}\\{}\\{}".format(basepath, label, file_name))
                        width, height = im.size    
                        
                        for xmin, ymin, xmax, ymax, label, score in zip(bboxes[0::6], bboxes[1::6], bboxes[2::6], bboxes[3::6], bboxes[4::6], bboxes[5::6]):
                                annotation = []
                                annotation.append(file_name.split("/")[1]) # Filename
                                annotation.append(width) # width
                                annotation.append(height) # height
                                annotation.append(labels[n]) # label
                                if int(xmin) > int(xmax):
                                    tmp = xmin
                                    xmin = xmax
                                    xmax = tmp
                                if int(ymin) > int(ymax):
                                    tmp = ymin
                                    ymin = ymax
                                    ymax = tmp
                                annotation.append(xmin) # xmin
                                annotation.append(ymin) # ymin
                                annotation.append(xmax) # xmax
                                annotation.append(ymax) # ymax
                                rows.append(annotation)

        column_name = ['image_id', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
        dataframe = pd.DataFrame(rows, columns=column_name)

        dataframe = shuffle(dataframe)
        
        dataframe.to_csv((basepath + '/test.csv'), index=None)

def GetCenter(xmin, ymin, xmax, ymax):
        x = int((int(xmax) + int(xmin)/2)) 
        y = int((int(ymax) + int(ymin)/2))
        return [x, y]

# # Create annotations for Yolo
#def createYolo(train_files):
#        rows = []
#        for n, file in enumerate(train_files):
#                for i, line in enumerate(open(file)):
#                        line_splitted = line.split() 
#                        file_name = line_splitted[0]
#                        label = labels[n]
#                        number_of_bboxes = int(line_splitted[1])
#                        bboxes = line_splitted[2:(number_of_bboxes*8)]
#                        im = Image.open("{}\\{}\\{}".format(basepath, label, file_name))
#                        width, height = im.size    
                        
#                        for xmin, ymin, xmax, ymax, label, score in zip(bboxes[0::6], bboxes[1::6], bboxes[2::6], bboxes[3::6], bboxes[4::6], bboxes[5::6]):
#                                x, y = GetCenter(xmin, ymin, xmax, ymax)
#                                print (x, y)
#                                break
                                
createCSV(train_files)