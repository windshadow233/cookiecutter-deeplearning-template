from models.model import Model
import os
from torch.utils.data import Dataset, random_split
from engine.trainer import Trainer
from engine.tester import Tester

from utils.config import load_end2end_cfg, get_config_value
from utils.misc import create_exp_dir
from utils.seed import set_seed
from utils.logger import init_logger, SimpleLogger
from metrics.classification import *
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset


class MyData(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = load_dataset("glue", "sst2", split=split).map(self.tokenize_fn)
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def tokenize_fn(self, item):
        return self.tokenizer(
            item['sentence'],
            padding='max_length',
            truncation=True,
            max_length=128
        )


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def forward(self, input_ids, attention_mask, label):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        return out

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path)


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

    def evaluate(self, model, dataloader):
        acc = Accuracy()
        recall = Recall()
        precision = Precision()
        f1 = F1Score()
        for batch in dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            acc.update(preds=preds, targets=batch['label'])
            recall.update(preds=preds, targets=batch['label'])
            precision.update(preds=preds, targets=batch['label'])
            f1.update(preds=preds, targets=batch['label'])
        return {
            'val_acc': acc.compute(),
            'val_recall': recall.compute(),
            'val_precision': precision.compute(),
            'val_f1': f1.compute()
        }


class MyTester(Tester):
    def __init__(self, model, ckpt_name, dataset, exp_dir, data_collate_fn=None):
        super().__init__(model, ckpt_name, dataset, exp_dir, data_collate_fn)

    def evaluate(self, model, dataloader):
        model.eval()
        acc = Accuracy()
        recall = Recall()
        precision = Precision()
        f1 = F1Score()
        for batch in dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            acc.update(preds=preds, targets=batch['label'])
            recall.update(preds=preds, targets=batch['label'])
            precision.update(preds=preds, targets=batch['label'])
            f1.update(preds=preds, targets=batch['label'])
        return {
            'test_acc': acc.compute(),
            'test_recall': recall.compute(),
            'test_precision': precision.compute(),
            'test_f1': f1.compute()
        }


if __name__ == "__main__":
    # load configuration
    cfg = load_end2end_cfg()
    # initialize logging
    init_logger(cfg)
    # create exp dir
    exp_dir = create_exp_dir(cfg, 'exp')
    # set seed
    seed = get_config_value(cfg, 'seed', 42)
    set_seed(seed)
    # create dataset
    dataset = MyData('train')
    trainset, validset = random_split(dataset=dataset, lengths=[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    testset = MyData('validation')
    # initialize model
    model = MyModel()
    # initialize trainer
    trainer = MyTrainer(
        model=model,
        train_dataset=trainset,
        valid_dataset=validset,
        cfg=cfg,
        exp_dir=exp_dir
    )
    trainer.run()
    if isinstance(trainer.logger, SimpleLogger):
        trainer.logger.plot(os.path.join(exp_dir, 'plot.png'))

    tester = MyTester(
        model=model,
        ckpt_name='best.pt',
        dataset=testset,
        exp_dir=exp_dir
    )
    tester.run()