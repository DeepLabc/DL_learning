import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET

def read_mnist(mode):
    """
    MNIST dataload
    """
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)]
    )
    trainset = torchvision.datasets.MNIST(root='./img', 
                                        train=True, 
                                        transform=transform, 
                                        download=True) # or False
    trainloader = DataLoader(dataset=trainset, batch_size=10, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./img', 
                                        train=False, 
                                        transform=transform, 
                                        download=True)
    testloader = DataLoader(dataset=testset, batch_size=10, shuffle=False, num_workers=4)

    if mode=='train':
        return trainloader
    elif mode=='test':
        return testloader
    else:
        print("mode error")


class myDataset(Dataset):
    """
    custom dataset
    """
    def __init__(self, data_path, label_path, img_size =(640,640), max_target=10,transform=None):
        self.class_to_ind = {'zebra':0}
        self.img_size = img_size
        self.max_target = max_target
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform

        assert len(os.listdir(self.data_path))== len(os.listdir(self.label_path))

        self.img_path = self.get_image_path()
        self.lab_path = self.get_label_path()

    def __getitem__(self, index):
        data = cv2.imread(self.img_path[index])
        r = min(self.img_size[0] / data.shape[0], self.img_size[1] / data.shape[1])
        data = cv2.resize(
            data,
            self.img_size,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        # Labels and image transforms should be consistent, here skip
        label = self.get_label(self.lab_path[index])
        label = np.pad(label,((0,self.max_target-label.shape[0]),(0,0)),'constant',constant_values =(0,0))
    
        return data, label

    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def get_image_path(self):
        img_path = []
        for p in os.listdir(self.data_path):
            img_path.append(os.path.join(self.data_path, p))
        img_path.sort()
        return img_path

    def get_label_path(self):
        lab_path = []
        for l in os.listdir(self.label_path):
            lab_path.append(os.path.join(self.label_path, l))
        lab_path.sort()
        return lab_path

    def get_label(self, lab):
        target = ET.parse(lab).getroot()
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]

            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res

if __name__ == "__main__":

    train_transform = transforms.Compose([
    transforms.Resize((700, 700)),
    transforms.RandomCrop(640,640),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

    dataset = myDataset('../img/imgs', '../img/anno')

    trainloader = DataLoader(dataset, batch_size=2,
                            shuffle=False, 
                            sampler=None, 
                            num_workers=2, 
                            collate_fn=None, 
                            pin_memory=False, 
                            drop_last=False)
    
    for b, (img, label) in enumerate(trainloader):
        print(img.shape)
        print(label.shape)