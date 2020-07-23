from spam.spam_classifier.datasets.dataset import Spam_img_dataset
from spam.spam_classifier.models.model import Classification_experiment
from spam.spam_classifier.networks.resnet50 import ResNet50

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': Classification_experiment,
    'fit_kwargs': {
        'batch_size': 128,
        'epochs_finetune': 3,
        'epochs_full': 3,
        'debug': False
    },
    'experiment_kwargs': {
        'network_fn': ResNet50,
        'network_kwargs': {},
        'dataset_cls': Spam_img_dataset,
        'name': "ResNet50",
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size,
            'rgb_mean' : [0.485, 0.456, 0.406], 
            'rgb_std' : [0.229, 0.224, 0.225]
        },
    },
}

