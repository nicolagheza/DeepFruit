from sklearn.utils import shuffle

import os
from PIL import Image
import pandas as pd

basepath = "C:\\Code\\DeepFruit\\datasets" 
train_files = []
labels = []
labels_number = [0,1,2,3,4,5,6]

for fruit in os.listdir(basepath):
        if fruit != 'avocado':
                continue
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
        
        dataframe.to_csv(('test.csv'), index=None)

def GetCenter(xmin, ymin, xmax, ymax):
        a = (int(xmin), int(ymin))
        b = (int(xmin), int(ymax))
        c = (int(ymax), int(ymax))
        d = (int(xmax), int(ymin))
        
        x = (a[0] + d[0])/2
        y = (a[1] + b[1])/2
        return (x, y)

def GetHeightWidth(xmin, ymin, xmax, ymax):
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
    return width, height

# # Create annotations for Yolo
def createYolo(train_files):
        rows = []
        index_file = open("test.txt", "w")
        for n, file in enumerate(train_files):
                for i, line in enumerate(open(file)):
                        line_splitted = line.split() 
                        file_name = line_splitted[0]
                        label = labels[n]
                        number_of_bboxes = int(line_splitted[1])
                        bboxes = line_splitted[2:(number_of_bboxes*8)] # Number of bounding boxes per file
                        im = Image.open("{}\\{}\\{}".format(basepath, label, file_name))
                        width, height = im.size    
                        bbox_count=0
                        for xmin, ymin, xmax, ymax, label, score in zip(bboxes[0::6], bboxes[1::6], bboxes[2::6], bboxes[3::6], bboxes[4::6], bboxes[5::6]):
                                if int(xmin) > int(xmax):
                                    tmp = xmin
                                    xmin = xmax
                                    xmax = tmp
                                if int(ymin) > int(ymax):
                                    tmp = ymin
                                    ymin = ymax
                                    ymax = tmp    
                                abs_x, abs_y = GetCenter(xmin, ymin, xmax, ymax)
                                abs_width, abs_height = GetHeightWidth(xmin, ymin, xmax, ymax)
                                x = abs_x / width
                                y = abs_y / height
                                abs_width = abs_width / width
                                abs_height = abs_height / height
                                im.save("images/{}-{}-{}.png".format(labels[n]+"-test-", i, bbox_count),'PNG')
                                file = open("data/{}-{}-{}.txt".format(labels[n]+"-test-", i,bbox_count),"w")
                                index_file.write("/content/gdrive/My Drive/darknet/img/{}-{}-{}.png\n".format(labels[n]+"-test-", i, bbox_count))                                
                                file.write("{} {} {} {} {}".format(labels_number[n], x, y, abs_width, abs_height))
                                file.close()
                                bbox_count = bbox_count + 1
                break
        index_file.close()
                                
# createYolo(train_files)
createCSV(train_files)
print("Finished.")