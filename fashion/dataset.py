# Import modules
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# Import PyTorch
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, json_path, transforms=None):
        self.data_path = data_path
        self.data_json = COCO(json_path)
        self.image_ids = list(self.data_json.imgToAnns.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        # Image path setting
        image_id = self.image_ids[idx]
        file_name = os.path.join(self.data_path, self.data_json.loadImgs(image_id)[0]['file_name'])

        # Image load
        image = Image.open(file_name).convert('RGB')
        annot_ids = self.data_json.getAnnIds(imgIds=image_id)
        annots = [x for x in self.data_json.loadAnns(annot_ids) if x['image_id'] == image_id]
        
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        masks = np.array([self.data_json.annToMask(annot) for annot in annots], dtype=np.uint8)

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd}
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)            

        return image, target

    def __len__(self):
        return len(self.image_ids)