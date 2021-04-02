import os
import cv2
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
    submit.to_csv('./submix.csv', index=False)