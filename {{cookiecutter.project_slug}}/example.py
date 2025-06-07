import os

import tqdm
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms as T
from models.model import Model
from engine.trainer import Trainer
from engine.tester import Tester

from utils.config import load_end2end_cfg, get_config_value
from utils.misc import create_exp_dir
from utils.seed import set_seed
from utils.logger import init_logger, SimpleLogger
from metrics.classification import *
from metrics.metric import AvgMetric


class MyData(Dataset):
    def __init__(self, train=True):
        super().__init__()
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return {'image': x, 'label': y}

    def __len__(self):
        return len(self.dataset)


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

    def forward(self, image, label):
        image = self.layer1(image)
        image = image.reshape(image.shape[0], -1)
        logits = self.layer2(image)
        loss = self.calc_loss(logits, label)
        return {
            'loss': loss,
            'logits': logits
        }

    def calc_loss(self, x, y):
        loss = self.loss_fcn(x, y)
        return loss


class MyTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataset,
            valid_dataset,
            cfg,
            exp_dir,
            data_collate_fn=None
    ):
        super().__init__(model, train_dataset, valid_dataset, cfg, exp_dir, data_collate_fn)

    @torch.no_grad()
    def evaluate(self, model, dataloader):
        f1 = F1Score()
        recall = Recall()
        precision = Precision()
        acc = Accuracy()
        loss = AvgMetric()
        for batch in tqdm.tqdm(dataloader, desc='Validation'):
            x, y = batch['image'], batch['label']
            count = len(y)
            out = model(x, y)
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
    def __init__(self, model, ckpt_name, dataset, exp_dir, data_collate_fn=None):
        super().__init__(model, ckpt_name, dataset, exp_dir, data_collate_fn)

    def evaluate(self, model, dataloader):
        f1 = F1Score()
        recall = Recall()
        precision = Precision()
        acc = Accuracy()
        loss = AvgMetric()
        for batch in tqdm.tqdm(dataloader, desc='Testing'):
            x, y = batch['image'], batch['label']
            count = len(y)
            out = model(x, y)
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
    # load configuration
    cfg = load_end2end_cfg()
    # initialize logging
    init_logger(cfg)
    # create exp dir
    exp_dir = create_exp_dir(cfg, 'exp')
    # or use a specific experiment directory
    # exp_dir = "runs/exp_20250607_091355"
    # set seed
    seed = get_config_value(cfg, 'seed', 42)
    set_seed(seed)
    # create dataset
    train_dataset = MyData(train=True)
    test_dataset = MyData(train=False)
    dataset_size = len(train_dataset)
    train_size, valid_size = int(0.8 * dataset_size), dataset_size - int(0.8 * dataset_size)
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    # initialize model
    model = MyModel()
    # initialize trainer
    trainer = MyTrainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        cfg=cfg,
        exp_dir=exp_dir
    )
    trainer.run()

    if isinstance(trainer.logger, SimpleLogger):
        trainer.logger.plot(os.path.join(exp_dir, 'plot.png'))

    tester = MyTester(
        model=model,
        ckpt_name='best.pt',
        dataset=test_dataset,
        exp_dir=exp_dir
    )
    tester.run()
