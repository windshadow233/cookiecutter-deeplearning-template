import os

import tqdm
from torch import nn
from torchvision import datasets, transforms as T
from models.model import Model
from engine.trainer import Trainer
from engine.tester import Tester

from utils.config import CONFIG
from utils.other_utils import create_exp_dir
from utils.seed import set_seed
from utils.logger import SimpleLogger
from metrics.classification import *
from metrics.metric import AvgMetric


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, dilation=(2, 2)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, dilation=(2, 2)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32 * 25, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
        self.loss_fcn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        loss = self.calc_loss(x, y)
        return {
            'loss': loss,
            'logits': x
        }

    def calc_loss(self, x, y):
        loss = self.loss_fcn(x, y)
        return loss


class MyTrainer(Trainer):
    def __init__(
            self,
            model,
            dataset,
            cfg,
            exp_dir
    ):
        super().__init__(model, dataset, cfg, exp_dir)

    def data_collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {'x': images, 'y': labels}

    @torch.no_grad()
    def validate_fn(self, dataloader):
        f1 = F1Score()
        recall = Recall()
        precision = Precision()
        acc = Accuracy()
        loss = AvgMetric()
        for batch in tqdm.tqdm(dataloader, desc='Validation'):
            x, y = batch['x'], batch['y']
            count = len(y)
            out = self.model(x, y)
            logits = out['logits']
            preds = torch.argmax(logits, dim=-1)
            f1.update(preds=preds, targets=y)
            recall.update(preds=preds, targets=y)
            precision.update(preds=preds, targets=y)
            acc.update(preds=preds, targets=y)
            loss.update(value=out['loss'], count=count)
        avg_f1 = f1.compute()
        avg_recall = recall.compute()
        avg_precision = precision.compute()
        avg_acc = acc.compute()
        avg_loss = loss.compute()

        return {
            'val_loss': avg_loss,
            'val_f1': avg_f1,
            'val_recall': avg_recall,
            'val_precision': avg_precision,
            'val_acc': avg_acc
        }


class MyTester(Tester):
    def __init__(self, model, ckpt_name, dataloader, exp_dir):
        super().__init__(model, ckpt_name, dataloader, exp_dir)

    def test_fn(self, dataloader):
        f1 = F1Score()
        recall = Recall()
        precision = Precision()
        acc = Accuracy()
        loss = AvgMetric()
        for batch in tqdm.tqdm(dataloader, desc='Testing'):
            x, y = batch['x'], batch['y']
            count = len(y)
            out = self.model(x, y)
            logits = out['logits']
            preds = torch.argmax(logits, dim=-1)
            f1.update(preds=preds, targets=y)
            recall.update(preds=preds, targets=y)
            precision.update(preds=preds, targets=y)
            acc.update(preds=preds, targets=y)
            loss.update(value=out['loss'], count=count)

        avg_f1 = f1.compute()
        avg_recall = recall.compute()
        avg_precision = precision.compute()
        avg_acc = acc.compute()
        avg_loss = loss.compute()

        return {
            'test_loss': avg_loss,
            'test_f1': avg_f1,
            'test_recall': avg_recall,
            'test_precision': avg_precision,
            'test_acc': avg_acc
        }


if __name__ == "__main__":
    # create exp dir
    exp_dir = create_exp_dir(CONFIG, 'exp')
    # or use a specific experiment directory
    # exp_dir = "runs/exp_20250604_152711"
    # set seed
    seed = CONFIG.get('seed', 42)
    set_seed(seed)
    # initialize model
    model = MyModel()
    # create dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
        T.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    # initialize trainer
    trainer = MyTrainer(
        model=model,
        dataset=train_dataset,
        cfg=CONFIG,
        exp_dir=exp_dir
    )
    trainer.run()

    if isinstance(trainer.logger, SimpleLogger):
        trainer.logger.plot(os.path.join(exp_dir, 'plot.png'))

    tester = MyTester(
        model=model,
        ckpt_name='best.pt',
        dataloader=trainer.test_dataloader,
        exp_dir=exp_dir
    )
    tester.run()
