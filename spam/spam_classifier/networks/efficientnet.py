from efficientnet_pytorch import EfficientNet

def EfficientNet_B3():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    print("EfficientNet B3 Loaded!")

    return model