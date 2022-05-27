import os
import random

image_path = "JPEGImages/"
xml_path = "Annotations/"

train_val_txt_path = "ImageSets/Main/"
val_percent = 0.2

image_list = os.listdir(image_path)
random.shuffle(image_list)

train_image_count = int((1-val_percent)*len(image_list))
val_image_count = int(val_percent*len(image_list))

train_txt = open(os.path.join(train_val_txt_path, "train.txt"), "w")
train_count = 0

for i in range(train_image_count):
    if image_list[i].split(".")[-1] == "jpg":
        text = image_list[i].split(".jpg")[0]+"\n"
    train_txt.write(text)
    train_count+=1
train_txt.close()

val_txt = open(os.path.join(train_val_txt_path, "val.txt"), "w")
val_count = 0
for i in range(val_image_count):
    #text = image_list[train_image_count+i].split(".jpg")[0] + "\n"
    if image_list[i].split(".")[-1] == "jpg":
        text = image_list[i].split(".jpg")[0]+"\n"
    val_count +=1
    val_txt.write(text)
val_txt.close()

