import torchvision
import PIL
import torch
rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

grid = {
    'batch_size' : [128, 64, 32],
    'class_ratio' : [
            [0.4, 0.2, 0.2, 0.2],
            [0.3, 0.2, 0.2, 0.3], 
            [0.25, 0.25, 0.25, 0.25]
        ],
    'epoch' : [
            [2,2], [1,3], [3,1], [4,1], [1,4]
        ],
    'gamma' : [0.3, 0.1],
    'learning_rate' : [1e-4, 1e-6, 1e-5],
    'loss_cls_weight' : [
            torch.Tensor([1,1,1,1]),
            torch.Tensor([1.006,1.0797,1.0402,1.2753]) # reciprocal of f-1 score of baseline
        ],
    'optimizer' : ['Adamax','Adam','SGD'], # ,'RMSprop'
    'step_size': [10, 2],
    'transforms' : [
            [1, torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])],
            [2, torchvision.transforms.Compose([
                torchvision.transforms.RandomAffine(0, translate=None, scale=(0.75, 1.25), shear=None, resample=False, fillcolor=0),
                torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=0.4, interpolation=3, fill=0),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])]
        ],
    'weight_decay' : [0, 1e-3]
}