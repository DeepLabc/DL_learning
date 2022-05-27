from email.mime import image
from logging import root
import os
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

HRSC_CLASSES={
"100000001": "ship",
"100000002": "aircraft carrier",
"100000003": "warcraft",
"100000004": "merchant ship",
"100000005": "Nimitz",
"100000006": "Enterprise",
"100000007": "Arleigh Burke",
"100000008": "WhidbeyIsland",
"100000009": "Perry",
"100000010": "Sanantonio",
"100000011": "Ticonderoga",
"100000012": "Kitty Hawk",
"100000013": "Kuznetsov",
"100000014": "Abukuma",
"100000015": "Austen",
"100000016": "Tarawa",
"100000017": "Blue Ridge",
"100000018": "Container",
"100000019": "OXo|--)",
"100000020": "Car carrier([]==[])",
"100000022": "Hovercraft",
"100000024": "yacht",
"100000025": "CntShip(_|.--.--|_]=",
"100000026": "Cruise",
"100000027": "submarine",
"100000028": "lute",
"100000029": "Medical",
"100000030": "Car carrier(======|",
"100000031": "Ford-class",
"100000032": "Midway-class",
"100000033": "Invincible-class"
}

# root_path = "/home/lyg/Documents/FullDataSet/Annotations"
# save_path = "/home/lyg/workspace/YOLOX_Det/datasets/HRSC2016/Annotations"  #obb
# img_path = "/home/lyg/workspace/YOLOX_Det/datasets/HRSC2016/IPEGImages"
# os.makedirs(save_path,exist_ok=True)

# obb_list = os.listdir(root_path)

# for o in obb_list:
#     doc = Document()
#     anno = doc.createElement('annotation')
#     doc.appendChild(anno)

#     folder = doc.createElement('folder')
#     anno.appendChild(folder)  
#     folder_txt = doc.createTextNode("VOC2007")  
#     folder.appendChild(folder_txt)


#     print("Processing {}".format(o))
#     xml_f = os.path.join(root_path, o)
#     tree = ET.parse(xml_f)
#     root = tree.getroot()
#     imgname = root.find('Img_FileName')

#     filename = doc.createElement('filename')
#     anno.appendChild(filename)
#     filename_txt = doc.createTextNode(img_path+'/'+imgname.text+'.bmp')
#     filename.appendChild(filename_txt)

#     w = root.find('Img_SizeWidth').text
#     h = root.find('Img_SizeHeight').text
#     d = root.find('Img_SizeDepth').text

#     size = doc.createElement('size')  
#     anno.appendChild(size) 

#     width = doc.createElement('width')  
#     size.appendChild(width)  
#     width_txt = doc.createTextNode(str(w))  
#     width.appendChild(width_txt) 

#     height = doc.createElement('height')  
#     size.appendChild(height)  
#     height_txt = doc.createTextNode(str(h))  
#     height.appendChild(height_txt)  
  
#     depth = doc.createElement('depth') 
#     size.appendChild(depth)  
#     depth_txt = doc.createTextNode(str(d))  
#     depth.appendChild(depth_txt) 

#     segmented = doc.createElement('segmented')  
#     anno.appendChild(segmented)  
#     segmented_txt = doc.createTextNode("0")  
#     segmented.appendChild(segmented_txt) 

#     objs = root.find('HRSC_Objects')
#     if not objs.find("HRSC_Object"):
#         continue
#     pts = ["box_xmin","box_ymin","box_xmax","box_ymax"]
#     for obj in objs.iter("HRSC_Object"):
       
#         object_new = doc.createElement("object")  
#         anno.appendChild(object_new) 
       
#         cls = HRSC_CLASSES[obj.find("Class_ID").text]
       
#         name = doc.createElement('name')  
#         object_new.appendChild(name)  
#         name_txt = doc.createTextNode(cls)  
#         name.appendChild(name_txt)

#         pose = doc.createElement('pose')  
#         object_new.appendChild(pose)  
#         pose_txt = doc.createTextNode("Unspecified")  
#         pose.appendChild(pose_txt)

#         truncated = doc.createElement('truncated')  
#         object_new.appendChild(truncated)  
#         truncated_txt = doc.createTextNode("0")  
#         truncated.appendChild(truncated_txt)  

#         difficult = doc.createElement('difficult')  
#         object_new.appendChild(difficult)  
#         difficult_txt = doc.createTextNode("0")  
#         difficult.appendChild(difficult_txt) 
         
#         bndbox = doc.createElement('bndbox')  
#         object_new.appendChild(bndbox) 

#         xmin_num = obj.find(pts[0]).text
#         xmin = doc.createElement('xmin')  
#         bndbox.appendChild(xmin)  
#         xmin_txt = doc.createTextNode(str(xmin_num))
#         xmin.appendChild(xmin_txt) 

#         ymin_num = obj.find(pts[1]).text
#         ymin = doc.createElement('ymin')  
#         bndbox.appendChild(ymin)  
#         ymin_txt = doc.createTextNode(str(ymin_num))
#         ymin.appendChild(ymin_txt)  

#         xmax_num = obj.find(pts[2]).text
#         xmax = doc.createElement('xmax')  
#         bndbox.appendChild(xmax)  
#         xmax_txt = doc.createTextNode(str(xmax_num))
#         xmax.appendChild(xmax_txt)  

#         ymax_num = obj.find(pts[3]).text
#         ymax = doc.createElement('ymax')  
#         bndbox.appendChild(ymax)  
#         ymax_txt = doc.createTextNode(str(ymax_num))
#         ymax.appendChild(ymax_txt) 
#     xml_name = o.split('.')[0]
#     full_path = os.path.join(save_path,xml_name+'.xml') 
#     with open(full_path, 'wb') as f:
#         f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

import os
from PIL import Image

# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = fileName.split('.')[0]+".jpg"
        print(newFileName)
        im = Image.open(file_path+"/"+fileName)
        im.save(file_path+"/"+newFileName)


# 删除原来的位图
def deleteImages(file_path, imageFormat):
    command = "rm -r "+file_path+"/*."+imageFormat
    os.system(command)


def main():
    file_path = "./JPEGImages"
    #bmpToJpg(file_path)
    deleteImages(file_path, "bmp")


if __name__ == '__main__':
    main()

