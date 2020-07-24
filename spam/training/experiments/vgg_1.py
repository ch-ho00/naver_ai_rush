from spam.spam_classifier.datasets.dataset import Spam_img_dataset
from spam.spam_classifier.models.model import Classification_experiment
from spam.spam_classifier.networks.vgg import VGG16
from spam.training.experiments.best_params import best_param

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': Classification_experiment,
    'fit_kwargs': [grid,best_param],
    'experiment_kwargs': {
        'network_fn': VGG16,
        'network_kwargs': {},
        'dataset_cls': Spam_img_dataset,
        'name': "VGG16",
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}
