from spam.spam_classifier.datasets.dataset import Spam_img_dataset
from spam.spam_classifier.models.model import Classification_experiment
from spam.spam_classifier.networks.efficientnet import EfficientNet_B3
from spam.training.experiments.grid_params import grid
from spam.training.experiments.best_params import best_param

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': Classification_experiment,
    'fit_kwargs': [grid, best_param],
    'experiment_kwargs': {
        'network_fn': EfficientNet_B3,
        'network_kwargs': {},
        'dataset_cls': Spam_img_dataset,
        'name': "EfficientNet_b3",
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}

