import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
from pycocotools.coco import COCO
from PIL import Image


class CustomCOCODataset(Dataset):
    def __init__(self,
                 root_folder,
                 annotation_json
    ):

        self.root = root_folder
        self.coco = COCO(annotation_json)
        self.cat_ids = self.coco.getCatIds(['dog','cat','horse','sheep','cow','bird','elephant','zebra','giraffe','bear'])
        id_list = torch.hstack([self.coco.getImgIds(catIds=[idx]) for idx in self.cat_ids]) #only include images of animals
        id_list_uniq = list(sorted(set(id_list))) #only include each image once
        id_list_3 = [i for i in id_list_uniq if self.check_channels(i)] #only include color images
        self.ids = id_list_3

    def __getitem__(self, 
                    index
    ):
        img_id = self.ids[index]
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_id)

        path = self.coco.loadImgs([img_id])[0]["file_name"]
        img = Image.open(os.path.join(self.root,path))
        n_objs = len(anns)

        # Bounding boxes for all objects in image
        # In coco format bbox = [xmin,ymin,width,height]
        # In pytorch, bbox = [xmin,ymin,xmax,ymax]
        boxes = []
        iscrowd = []
        for i in range(n_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]
            boxes.append([xmin,ymin,xmax,ymax])
            iscrowd.append(anns[i["iscrowd"]])
        
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd,dtype=torch.float32)
        labels = torch.ones((n_objs,),dtype=torch.int64)
        img_id = torch.tensor([img_id])

        img_tensor, scaled_boxes = self.prepare_image(img_tensor,boxes)

        my_annotations = {}
        my_annotations["image_id"] = img_id
        my_annotations["path"] = path
        my_annotations["boxes"] = scaled_boxes
        my_annotations["labels"] = labels
        my_annotations["iscrowd"] = iscrowd

        return img_tensor, my_annotations







