

from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import random
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str)
    parser.add_argument('-p', '--percent', type=float, default=0.05)
    parser.add_argument('-t', '--train', type=str, default='train.csv')
    parser.add_argument('-v', '--val', type=str, default='val.csv')
    parser.add_argument('-c', '--classes', type=str, default='class.csv')
    args = parser.parse_args()
    return args

#获取特定后缀名的文件列表
def get_file_index(indir, postfix):
    file_list = []
    for root, dirs, files in os.walk(indir):
        for name in files:
            if postfix in name:
                file_list.append(os.path.join(root, name))
    return file_list

#写入标注信息
def convert_annotation(csv, address_list):
    cls_list = []
    with open(csv, 'w') as f:
        for i, address in enumerate(address_list):
            in_file = open(address, encoding='utf8')
            strXml =in_file.read()
            in_file.close()
            root=ET.XML(strXml)
            for obj in root.iter('object'):
                cls = obj.find('name').text
                cls_list.append(cls)
                xmlbox = obj.find('bndbox')
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                     int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                f.write(file_dict[address_list[i]])
                f.write( "," + ",".join([str(a) for a in b]) + ',' + cls)
                f.write('\n')
    return cls_list


if __name__ == "__main__":
    args = parse_args()
    file_address = args.indir
    test_percent = args.percent
    train_csv = args.train
    test_csv = args.val
    class_csv = args.classes

    train_xml = []
    train_img = []
    for line in open('/home/lyg/workspace/YOLOX_Det/datasets/animal/ImageSets/Main/train.txt'):
        train_xml.append('/home/lyg/workspace/YOLOX_Det/datasets/animal/Annotations/'+line.strip()+'.xml')
        train_img.append('/home/lyg/workspace/YOLOX_Det/datasets/animal/JPEGImages/'+line.strip()+'.jpg')

    test_xml = []
    test_img = []
    for line in open('/home/lyg/workspace/YOLOX_Det/datasets/animal/ImageSets/Main/val.txt'):
        test_xml.append('/home/lyg/workspace/YOLOX_Det/datasets/animal/Annotations/'+line.strip()+'.xml')
        test_img.append('/home/lyg/workspace/YOLOX_Det/datasets/animal/JPEGImages/'+line.strip()+'.jpg')

    Annotations = train_xml + test_xml
    JPEGfiles = train_img + test_img
    Annotations.sort()
    JPEGfiles.sort()

    assert len(Annotations) == len(JPEGfiles) #若XML文件和图片文件名不能一一对应即报错
    file_dict = dict(zip(Annotations, JPEGfiles))
    num = len(Annotations)
    # test = random.sample(k=math.ceil(num*test_percent), population=Annotations)
    # train = list(set(Annotations) - set(test))

    cls_list1 = convert_annotation(train_csv, train_xml)
    cls_list2 = convert_annotation(test_csv, test_xml)
    cls_unique = list(set(cls_list1+cls_list2))

    with open(class_csv, 'w') as f:
        for i, cls in enumerate(cls_unique):
            f.write(cls + ',' + str(i) + '\n')
