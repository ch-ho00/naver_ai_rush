from typing import Callable, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
import pandas as pd
import nsml
import numpy as np

# from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate
from spam.spam_classifier.datasets.dataset import Spam_img_dataset


class Classification_experiment:
    
    def __init__(self, network_fn: Callable, dataset_cls: Spam_img_dataset, dataset_kwargs, network_kwargs, name="ResNet50",to_log=None, prefix=None):
        self.data = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))
        self.model = network_fn(**kwargs_or_empty_dict(network_kwargs))
        self.debug = False
        self.name = name

        if to_log is None:
            to_log = ['loss', 'accuracy']
        self.to_log = to_log + [f'val_{key}' for key in to_log]
        self.prefix = prefix


    def fit(self, epochs_finetune, epochs_full, batch_size, lr=1e-4, weight_decay=0, gamma=0.3, step_size=10, debug=False):
        self.debug = debug
        self.data.prepare()
        self.model = self.model.cuda()

        steps_per_epoch_train = int(self.data.len('train') / batch_size) if not self.debug else 2
        trainloader, validationloader = self.data.train_val_gen(batch_size)

        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # train the model by epochs
        self.train(epochs_finetune,trainloader, validationloader, scheduler, optimizer, criterion)
        self.unfreeze()
        self.train(epochs_full,trainloader, validationloader, scheduler, optimizer, criterion, epochs_finetune)
        print("---train finished---")
        nsml.save(checkpoint='best')
        print('Done')
        # self.metrics(gen=validationloader)

    def train(self,epochs, trainloader, validationloader,scheduler, optimizer, criterion,prev_epoch=None):

        batch_size = trainloader.batch_size
        for epoch in range(epochs):
            loss_sum = 0
            val_loss_sum = 0
            correct  = 0
            for iter_, (xx, yy) in enumerate(trainloader):
                xx = xx.cuda()
                yy = yy.squeeze(1).long().cuda()
                
                if "Dense" in self.name or "VGG" in self.name:
                    xx = F.interpolate(xx,(224,224))

                optimizer.zero_grad()
                outputs = self.model(xx)
                batch_correct = self.count_correct(outputs, yy) 
                correct += batch_correct

                loss = criterion(outputs, yy)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

                print("Iteration %d : batch_loss = %2.5f, accuracy = %1.5f"%(iter_+1, loss, (batch_correct/batch_size)))

            val_outputs = []
            val_ytrue = []
            val_correct = 0
            with torch.no_grad():
                for valX, valY in validationloader:
                    valX = valX.cuda()
                    valY = valY.squeeze(1).long().cuda()

                    if "Dense" in self.name or "VGG" in self.name:
                        valX = F.interpolate(valX,(224,224))

                    outputs = self.model(valX)

                    val_correct += self.count_correct(outputs, valY) 
                    val_loss = criterion(outputs, valY)
                    val_loss_sum += val_loss

                    val_outputs.append(outputs)
                    val_ytrue.append(valY)

            scheduler.step()
            acc = correct / len(trainloader.dataset)
            val_acc = val_correct / len(validationloader.dataset)
            if prev_epoch:
                epoch += prev_epoch
            print("Epoch: %d, batch loss: %1.5f Loss Sum: %1.5f, accuracy : %1.5f, val loss: %1.5f, val accuracy: %1.5f"% (epoch, loss.item(), loss_sum, acc, val_loss_sum, val_acc))
            
            logs = {
                'val_loss' : val_loss_sum.item(),
                'val_accuracy': val_acc,
                'loss': loss_sum,
                'accuracy': acc
            }
                # 'lr': optimizer.default['lr']
            val_outputs = torch.cat(val_outputs, axis=0)
            _, val_outputs = torch.max(val_outputs,axis=1)
            val_outputs = val_outputs.long()

            val_ytrue = torch.cat(val_ytrue, axis=0)
            self.log(epoch, val_outputs, val_ytrue, logs)

    def log(self, epoch, val_pred_class, val_true_class, logs=None):

        cls_report = classification_report(
            y_true=val_true_class.detach().cpu().numpy(),
            y_pred=val_pred_class.detach().cpu().numpy(),
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )
        # log confusion matrix detail
        for label, res in cls_report.items():
            if isinstance(res, dict):
                to_report = {f'val__{self.name}/{label}/{k}': v for k, v in res.items()}
            else:
                to_report = {f'val__{self.name}/{label}': res}

            nsml.report(step=epoch, **to_report)

            # NOTE the val__ is needed for the nsml plotting.
            for k, v in to_report.items():
                logs[k.replace('val__', 'val/')] = v

        # log loss
        nsml_keys = []
        for k in self.to_log:
            if 'val' in k:
                k_ = k.replace("val_", "val__")
            else:
                k_ = f'train__{k}'
            if self.prefix is not None:
                nsml_keys.append(f'{self.prefix}__{k_}')
            else:
                nsml_keys.append(k_)


        nsml.report(
            step=epoch,
            **{nsml_key: self._to_json_serializable(logs.get(k)) for nsml_key, k
               in zip(nsml_keys, self.to_log)}
        )
        # log final score 
        f1_keys = ['val/'+ self.name + '/' +class_+ '/f1-score' for class_ in self.data.classes][1:]
        score = np.prod([logs[key] for key in f1_keys]) ** (1/3)
        
        print("Final :", score, [(key_, val_) for key_, val_ in zip(f1_keys,[logs[key] for key in f1_keys])])
        nsml.report(step=epoch,
            **{'Score': score}
        )
    def format_test(self, test_dir: str) -> pd.DataFrame:
        """

        Args:
            test_dir: Path to the test dataset.

        Returns:
            ret: A dataframe with the columns filename and y_pred. One row is the prediction (y_pred)
                for that file (filename). It is important that this format is used for NSML to be able to evaluate
                the model for the leaderboard.

        """
        gen, filenames = self.data.test_gen(test_dir=test_dir, batch_size=64)
        y_pred = self.test(gen)
        # y_pred = self.model.predict_generator(gen)

        ret = pd.DataFrame({'filename': filenames, 'y_pred': np.argmax(y_pred, axis=1)})
        return ret

    def test(self, testloader):
        ypreds = []
        for xx in testloader:
            xx = xx.cuda()

            if "Dense" in self.name or "VGG" in self.name:
                valX = F.interpolate(valX,(224,224))

            ypred = self.model(xx)
            ypreds.append(ypred)
        return torch.cat(ypreds,axis=0).detach().numpy()

    def evaluate(self, data_gen):
        y_pred = []
        y_true = []
        with torch.no_grad():
            for xx, yy in data_gen:
                xx = xx.cuda()
                yy = yy.squeeze(1).long().cuda()

                if "Dense" in self.name or "VGG" in self.name:
                    valX = F.interpolate(valX,(224,224))

                ypred = self.model(xx)
                y_pred.append(ypred)
                y_true.append(yy)
        y_true = torch.cat(y_true,axis=0).detach().cpu().numpy()
        y_pred = torch.cat(y_pred,axis=0).detach().cpu().numpy()
        
        return y_true, y_pred

    def unfreeze(self) -> None:
        for name, params in self.model.named_parameters():
            params.requires_grad = True

    def count_correct(self, outputs, y):
        _, outputs = torch.max(outputs, axis=1)

        return (outputs == y).nonzero().shape[0]

    def fit_metrics(self) -> List[str]:
        return ['accuracy']

    def _to_json_serializable(self, v):
        return v if not isinstance(v, np.float32) else v.item()
 
    def metrics(self, gen) -> None:
        """
        Generate and print metrics.

        Args:
            gen: Keras generator for which to get metrics
            n_batches: How many batches that can be fetched from the data generator.
        """
        y_true, y_pred = self.evaluate(data_gen=gen)
        y_true, y_pred = [np.argmax(y, axis=1) for y in [y_true, y_pred]]

        cls_report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )
        print(
            f'Classification report for validation dataset:\n-----------------------------\n{cls_report}\n=============\n')


def bind_model(experiment: Classification_experiment):
    """
    Utility function to make the model work with leaderboard submission.
    """

    def load(dirname, **kwargs):
        experiment.model.load_state_dict(torch.load(f'{dirname}/model'))
        experiment.model.eval()

    def save(dirname, **kwargs):
        filename = f'{dirname}/model'
        print(f'Trying to save to {filename}')
        torch.save(experiment.model.state_dict(), f'{dirname}/model')

    def infer(test_dir, **kwargs):
        return experiment.format_test(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
    