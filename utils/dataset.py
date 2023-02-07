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
        # TODO: No need for the for loop
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
                      size
    ):
        #resizes and turns image to tensor, ensures boxes still lie on objects

        scaled_boxes = boxes
        scaled_boxes[:,[0,2]] = (size*(boxes[:,[0,2]]/img.size[0]))
        scaled_boxes[:,[1,3]] = (size*(boxes[:,[1,3]]/img.size[1]))

        resize_tens = transforms.Compose([transforms.Resize([size,size]),
                                       transforms.ToTensor()])
        scaled_img = resize_tens(img)
        return scaled_img, scaled_boxes


    def check_channels(self,
                      img_id
    ):
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

    
    def custom_collate_fn(self,
                          batch
    ):
    
    # Function to correctly stack images/annotations inside the batch
    # Output: 
    # images: (batch_size, 3, 256, 256)
    # boxes: (batch_size, n_obj, 4)
    # labels: (batch_size, n_obj)
    # index: (batch_size)
    # path: (batch_size)

        img_tensor_list = []
        scaled_boxes_list = []
        labels_list = []
        index_list = []
        path_list = []

        for img_tensor, my_anns in batch:
            img_tensor_list.append(img_tensor)
            scaled_boxes_list.append(my_anns["boxes"])
            labels_list.append(my_anns["labels"])
            index_list.append(my_anns["image_id"])
            path_list.append(my_anns["path"])
        
        batch_images = torch.stack(img_tensor_list,dim=0)

        batch_anns = dict(bboxes = scaled_boxes_list,
                          labels = labels_list,
                          image_index = index_list,
                          image_paths = path_list,)

        return batch_images, batch_anns
    
    
    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)








