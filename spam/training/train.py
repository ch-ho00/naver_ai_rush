from importlib import import_module

import nsml

from spam.spam_classifier.models.model import bind_model


def train(experiment_name: str = 'resnet_1', pause: bool = False, mode: str = 'train'):
    config = import_module(f'spam.training.experiments.{experiment_name}').config
    model = config['model'](**config['experiment_kwargs'])
    bind_model(model)
    if pause:
        nsml.paused(scope=locals())
    if mode == 'train':
        model.fit(**config['fit_kwargs'])
