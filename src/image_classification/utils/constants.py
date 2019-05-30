import os
width = 100
height = 100
root_dir = os.getcwd()
data_dir = root_dir + '/data/' # 'C:\\Code\\DeepFruit\\experiments\\FruitDB\\data\\' 
fruit_models_dir =  root_dir + '/fruit_models/' #'C:\\Code\\DeepFruit\\experiments\\FruitDB\\fruit_models\\' 
labels_file = root_dir + '/utils/labels'

# change this to the path of the folders that hold the images
training_images_dir = 'C:/Code/DeepFruit/output/train'
test_images_dir = 'C:/Code/DeepFruit/output/test'

# number of classes: number of fruit classes + 1 resulted due to the build_image_data.py script that leaves the first class as a background class
# using the labels file that is also used in the build_image_data.py
num_classes = 15
number_train_images = 35518 
number_test_images = 8888
# with open(labels_file) as f:
#     labels = [line.rstrip() for line in f]
#     num_classes = len(labels) + 1
#     for label in labels:
#         number_train_images += len(os.listdir(os.path.join(training_images_dir, label)))
#         number_test_images += len(os.listdir(os.path.join(test_images_dir, label)))