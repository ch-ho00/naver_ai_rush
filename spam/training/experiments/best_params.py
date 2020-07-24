import torchvision
import PIL
import torch
rgb_mean = [0.485, 0.456, 0.406] 
rgb_std = [0.229, 0.224, 0.225]

best_param = {
    'batch_size' : [64],
    'class_ratio' : [
            [0.3, 0.2, 0.2, 0.3], 
        ],
    'epoch' : [
            [4,2]
        ],
    'gamma' : [0.1],
    'learning_rate' : [1e-4],
    'loss_cls_weight' : [
            torch.Tensor([1,1,1,1])
        ],
    'optimizer' : ['Adamax'], 
    'step_size': [10],
    'transforms' : [
            [1,None]
        ],
    'weight_decay' : [0]
}


            # [1, torchvision.transforms.Compose([
            #     torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            #     torchvision.transforms.RandomHorizontalFlip(),
            #     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            #     torchvision.transforms.ToTensor(),
            #     torchvision.transforms.Normalize(rgb_mean, rgb_std)
            # ])]
