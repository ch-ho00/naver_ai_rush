import torch
import torch.nn as nn
from spam.spam_classifier.networks.densenet import DenseNet121
from spam.spam_classifier.networks.resnet50 import ResNet50
from spam.spam_classifier.networks.vgg import VGG16
import xgboost as xgb

class Ensemble_model(nn.Module):
    def __init__(self, mode='soft', weight=None):
        super(Ensemble_model, self).__init__()
        self.densenet = DenseNet121()
        self.vgg = VGG16()
        self.resnet = ResNet50()
        
        self.weight = weight
        self.mode = mode
        if mode == 'stacked':
            self.stacked_fc = nn.Linear(4 * 3, 4)

    def forward(self, x):
        if self.mode == 'soft':
            y1= self.densenet(x.clone())
            y2= self.vgg(x.clone())
            y3= self.resnet(x)

            if self.weight == None:
                y = (y1 + y2 + y3) / 3
            else:
                w1, w2, w3 = self.weight
                y = y1 * w1 + y2* w2 + y3* w3
            return y

        elif self.mode == 'stacked':
            y1= self.densenet(x.clone())
            y2= self.vgg(x.clone())
            y3= self.resnet(x)

            ypred = torch.cat([y1, y2, y3], axis=1)
            ypred = self.stacked_fc(ypred)
            
            return ypred 
        # elif 

    def save(self):
        pickle.dump(model, open("pima.pickle.dat", "wb"))

    def load(self):
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
