from spam.spam_classifier.datasets.dataset import Spam_img_dataset
from spam.spam_classifier.models.model import Classification_experiment
from spam.spam_classifier.networks.ensemble_model import Ensemble_model
from spam.training.experiments.grid_params import grid
from spam.training.experiments.best_params import best_param

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': Classification_experiment,
    'fit_kwargs': [grid, best_param],
    'experiment_kwargs': {
        'network_fn': Ensemble_model,
        'network_kwargs': {
            'xgb': False,
            'std': [0,1,1],
            # order is vgg resnet densenet
            'pretrained': [['chanhopark00/spam-3/110','best_10'], ['chanhopark00/spam-3/109','best_0'],['chanhopark00/spam-3/111', 'best_1'], ['chanhopark00/spam-3/122','best_10']],
            'mode' : 'stacked'
        },
        'name': "Ensemble_model",
        'dataset_cls': Spam_img_dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}

