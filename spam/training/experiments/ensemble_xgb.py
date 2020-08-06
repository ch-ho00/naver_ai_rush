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
            'xgb': True,
            'std': [0,1,1],
            # order is vgg resnet densenet
            # top2
            # [['chanhopark00/spam-3/54','best_0'], ['chanhopark00/spam-3/40','best_4'], ['chanhopark00/spam-3/55','best_1']],
            # before changing validation set 
            # 'pretrained': [['chanhopark00/spam-3/54','best_1'], ['chanhopark00/spam-3/71','best_8'], ['chanhopark00/spam-3/73','best_5']],
            # before changing train dataset
            # 'pretrained': [['chanhopark00/spam-3/110','best_10'], ['chanhopark00/spam-3/109','best_0'],['chanhopark00/spam-3/111', 'best_1'], ['chanhopark00/spam-3/122','best_10']],
            # before considering normal unknown false negative
            # 'pretrained': [['chanhopark00/spam-3/159','best_1'], ['chanhopark00/spam-3/158','best_3'],['chanhopark00/spam-3/163','best_5'], ['chanhopark00/spam-3/122','best_0']],
            # 'pretrained': [['chanhopark00/spam-3/191','best_5'], ['chanhopark00/spam-3/192','best_6'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            # ________________________________ tried
            # [0,0,1]
            # 'pretrained': [['chanhopark00/spam-3/191','best_5'], ['chanhopark00/spam-3/192','best_6'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            # [0,1,1]
            # 'pretrained': [['chanhopark00/spam-3/54','best_0'], ['chanhopark00/spam-3/40','best_4'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            # [0,0,1]
            # 'pretrained': [['chanhopark00/spam-3/54','best_0'], ['chanhopark00/spam-3/192','best_6'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            # [0,1,1] 238
            # 'pretrained': [['chanhopark00/spam-3/191','best_5'], ['chanhopark00/spam-3/40','best_4'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            # [0,1,1] 239
            # 'pretrained': [['chanhopark00/spam-3/54','best_1'], ['chanhopark00/spam-3/40','best_4'],['chanhopark00/spam-3/163','best_5'], ['chanhopark00/spam-3/122','best_0']],
            # [0,1,1] 240
            # 'pretrained': [['chanhopark00/spam-3/54','best_1'], ['chanhopark00/spam-3/40','best_4'],['chanhopark00/spam-3/163','best_5'], ['chanhopark00/spam-3/122','best_0']],
            # [0,1,1] 241
            # 'pretrained': [['chanhopark00/spam-3/191','best_5'], ['chanhopark00/spam-3/40','best_4'],['chanhopark00/spam-3/194','best_9'], ['chanhopark00/spam-3/122','best_0']],
            
            'mode' : 'xgb'
        },
        'name': "Ensemble_model",
        'dataset_cls': Spam_img_dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}

