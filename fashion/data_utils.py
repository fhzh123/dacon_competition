import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(
                image, target)
            
        return image, target

class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        w, h = image.size
        image = image.resize(self.size)

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        
        for i, v in enumerate(_masks):
            v = Image.fromarray(v).resize(self.size, resample=Image.BILINEAR)
            masks[i] = np.array(v, dtype=np.uint8)

        target['masks'] = masks
        target['boxes'][:, [0, 2]] *= self.size[0] / w
        target['boxes'][:, [1, 3]] *= self.size[1] / h
        
        return image, target
        
class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        
        return image, target

class Normalize:
    def __call__(self, image, target):

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        image=TF.normalize(image,mean,std)

        return image, target