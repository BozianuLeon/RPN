import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import os
from pycocotools.coco import COCO
from PIL import Image


class CustomCOCODataset(Dataset):
    '''
    Preparing COCO images à la h-huang-github.io/tutorials/intermediate/torchvision_tutorial.html
    '''
    def __init__(self,
                 root_folder,
                 annotation_json
    ):

        self.root = root_folder
        self.coco = COCO(annotation_json)
        self.cat_ids = self.coco.getCatIds(['dog','cat','horse','sheep','cow','bird','elephant','zebra','giraffe','bear'])
        id_list = np.hstack([self.coco.getImgIds(catIds=[idx]) for idx in self.cat_ids]) #only include images of animals
        id_list_uniq = list(sorted(set(id_list))) #only include each image once
        id_list_3 = [i for i in id_list_uniq if self.check_channels(i)] #only include color images
        self.ids = id_list_3


    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_id)

        path = self.coco.loadImgs([img_id])[0]["file_name"]
        img = Image.open(os.path.join(self.root,path))
        n_objs = len(anns)

        # Bounding boxes for all objects in image
        # In coco format bbox = [xmin,ymin,width,height]
        # In pytorch, bbox = [xmin,ymin,xmax,ymax]
        # TODO: No need for the for loop
        boxes = []
        iscrowd = []
        for i in range(n_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]
            boxes.append([xmin,ymin,xmax,ymax])
            iscrowd.append(anns[i]["iscrowd"])
        
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd,dtype=torch.float32)
        labels = torch.ones((n_objs,),dtype=torch.int64)
        img_id = torch.tensor([img_id])

        img_tensor, scaled_boxes = self.prepare_image(img,boxes) # ensure all images are same size, and boxes still on objects

        my_annotations = {}
        my_annotations["image_id"] = img_id
        my_annotations["path"] = path
        my_annotations["boxes"] = scaled_boxes
        my_annotations["labels"] = labels
        my_annotations["iscrowd"] = iscrowd

        return img_tensor, my_annotations


    def __len__(self):
        return len(self.ids)
    

    def prepare_image(self,
                      img,
                      boxes,
                      size = (256,256),
    ):
        #resizes and turns image to tensor, ensures boxes still lie on objects

        scaled_boxes = boxes
        scaled_boxes[:,[0,2]] = (size[0]*(boxes[:,[0,2]]/img.size[0]))
        scaled_boxes[:,[1,3]] = (size[1]*(boxes[:,[1,3]]/img.size[1]))

        resize_tens = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor()])
        scaled_img = resize_tens(img)
        return scaled_img, scaled_boxes


    def check_channels(self, img_id):
        #checks that we have an RGB image
        img_path = self.coco.loadImgs([img_id])[0]["file_name"]
        im = Image.open(os.path.join(self.root, img_path))
        if len(im.mode)==1:
            im.close()
            return False
        elif len(im.mode)==3:
            im.close()
            return True
        else:
            im.close()
            return -1






class CustomToyDataset(Dataset):
    '''
    Preparing toy data dataset, result in same format as COCO dataset
    '''
    def __init__(self,annotation_json):

        with open(annotation_json) as json_file:
            annotations = json.load(json_file)
        self.annotations = annotations
        self.ids = torch.arange(len(self.annotations))

    def __getitem__(self, index):
        annotations_i = self.annotations[str(index)]
        path = annotations_i["image"]["img_path"]
        img = Image.open(path).convert('L')
        img.show()
        print(img.mode)

        n_objs = annotations_i["annotations"]["n_gausses"]

        tensor_boxes = torch.as_tensor(annotations_i["annotations"]["bboxes"])
        boxes = self.xywh2xyxy(tensor_boxes)
        labels = torch.ones((n_objs,),dtype=torch.int32)

        img_tensor, scaled_boxes = self.prepare_image(img,boxes)

        my_annotations = {}
        my_annotations["image_id"] = index
        my_annotations["path"] = path
        my_annotations["boxes"] = scaled_boxes
        my_annotations["labels"] = labels

        return img_tensor, my_annotations
    
    def __len__(self):
        return len(self.ids)

    def xywh2xyxy(self,x):
        x, y, w, h = x.unbind(-1)
        b = [x,y,x+w,y+h]
        return torch.stack(b, dim=-1)
    
    def prepare_image(self,
                      img,
                      boxes,
                      size=(256,256)
    ):
        
        scaled_boxes = boxes
        scaled_boxes[:,[0,2]] = (size[0]*(boxes[:,[0,2]]/img.size[0]))
        scaled_boxes[:,[1,3]] = (size[1]*(boxes[:,[1,3]]/img.size[1]))   

        resize_tens = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor()])
        scaled_img = resize_tens(img)
        return scaled_img, scaled_boxes








class CustomCOCODataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 num_workers
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    
    def custom_collate_fn(self, batch):

        # Function to correctly stack images/annotations inside the batch
        # Output: 
        # images: (batch_size, 3, 256, 256)
        # targets: List[Dict[str, Tensor]]

        # boxes: (batch_size, n_obj, 4)
        # labels: (batch_size, n_obj)
        # index: (batch_size)
        # path: (batch_size)

        img_tensor_list = []
        batch_targets_list = []
        for img_tensor, my_anns in batch:
            img_tensor_list.append(img_tensor)
            targets = dict(
                boxes=my_anns['boxes'],
                labels=my_anns['labels'],
                image_index=my_anns["image_id"],
                image_path=my_anns['path']
            )
            batch_targets_list.append(targets)
                
        batch_images = torch.stack(img_tensor_list,dim=0)

        return batch_images, batch_targets_list
    
    
    def loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)








