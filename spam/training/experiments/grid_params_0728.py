import torchvision
import PIL
import torch
rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

grid = {
    'batch_size' : [128],
    'class_ratio' : [
            [0.35, 0.05, 0.15, 0.45],
            [0.4, 0.1, 0.1, 0.4]
        ],
    'epoch' : [
            [5,10], [0,15]
        ],
    'gamma' : [0.1, 0.3],
    'learning_rate' : [1e-4, 1e-5],
    'loss_cls_weight' : [
            torch.Tensor([1,1,1,1]),
            torch.Tensor([1,1,1,1.1]),
            torch.Tensor([1,1,1,1.2])

        ],
    'optimizer' : ['Adam'], # 
    'step_size': [5, 2],
    'transforms' : [
            [1, None],
            [2, torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])]
        ],
    'val_ratio' : [
            # [0.02, 0.2, 0.2, 0.25],
            # [0.01, 0.15, 0.15, 0.25]
            [0.005, 0.25, 0.2, 0.2]
            # [0.02, 0.2, 0.15, 0.25]
        ],
    'weight_decay' : [0, 1e-3]
}