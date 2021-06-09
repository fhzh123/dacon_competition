import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def testing(args):

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Submit data open
    submit = pd.read_csv('/HDD/dataset/dacon/pose/sample_submission.csv')

    # Model setting
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )
    model = KeypointRCNN(
        backbone, 
        num_classes=2,
        num_keypoints=24,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler
    )
    model = model.to(device)

    checkpoint = torch.load(args.file_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model = model.eval()

    for i, img_id in enumerate(tqdm(submit['image'])):
        image = cv2.imread(os.path.join('/HDD/dataset/dacon/pose/test_imgs/', img_id), cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = [torch.as_tensor(image, dtype=torch.float32).to(device)]


        preds = model(image)
        preds_ = preds[0]['keypoints'][0][:,:2].detach().cpu().numpy().reshape(-1)
        submit.iloc[i, 1:] = preds_

    # Save
    error_list = np.array([
        317, 869, 873, 877, 911, 1559, 1560, 1562, 1566, 1575, 1577, 1578, 1582, 1606, 1607, 1622, 1623, 
        1624, 1625, 1629, 3968, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 
        4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 
        4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 
        4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 
        4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 
        4191, 4192, 4193, 4194
    ])
    submit2 = submit.loc[~submit.index.isin(error_list)]
    submit.to_csv('./submix_new.csv', index=False)
    submit2.to_csv('./submix_new2.csv', index=False)