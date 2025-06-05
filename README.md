# 一个还在构建中的深度学习项目模板

以往做深度学习项目时，每次新建项目总得从头到尾写一整套框架，其中不乏一些重复累赘的代码，比如超参数配置，训练、测试流程，以及指标计算、日志记录等等。
感觉非常繁琐，遂想着能不能将一些常用的功能进行集成，在需要用到时能够开箱即用。虽然已经有了很多现成的工具和框架，但有的过于复杂，有的则缺少一些特定的功能，例如断点续训等。

考虑到这些问题，我决定从头开始构建一个深度学习项目模板，能够让我自己快速上手并满足相当一部分深度学习项目的需求。于是就有了这个仓库。

由于写的比较仓促，可能会有很多考虑不到的需求，以及一堆 bug，不过主要是自己使用，暂时不考虑太多，等遇上了再逐项添加、解决。

## 主要功能

- **超参数配置**：使用 YAML 文件进行超参数配置。
- **训练流程**：提供了一个较为通用的单目标优化训练器，支持断点续训。
- **测试流程**：提供了一个较为通用的测试器，支持多种评估指标。
- **日志记录**：支持使用多种日志记录工具进行训练过程的可视化。
- **测试指标**：封装了多种常见任务场景下的常用测试指标计算，如准确率、精确率、召回率等。

## 使用方法

项目包含一个示例脚本 `example.py`，可以作为使用模板的参考。以下是一些基本的使用步骤：

新建项目环境，然后安装 `cookiecutter`：

```bash
pip install cookiecutter
````

使用 `cookiecutter` 命令通过此模板创建新项目：

```bash
cookiecutter git@github.com:windshadow233/cookiecutter-deeplearning-template.git
```

进入项目目录，安装依赖：

```bash
chmod +x setup.sh
./setup.sh
```

然后即可开始深度学习项目开发。

---

主要流程如下：

首先，定义数据集，继承自 `dataset.dataset.Dataset` 类，并实现必要的方法：

- `__getitem__`：获取单个数据样本。
- `__len__`：返回数据集的大小。
- `data_collate_fn`：用于将一批数据样本合并成一个批次，通常用于处理不同大小的输入数据，返回一个 `dict`。

默认的 `data_collate_fn` 方法会将数据样本的第一个元素作为输入特征 `x`，第二个元素作为标签 `y`，并将它们打包成一个字典返回。你可以根据需要自定义这个方法来处理更复杂的数据结构。

`data_collate_fn` 方法必须返回一个 `dict`，其键值将作为模型前向传播函数的`**kwargs`使用。（如重写 `Trainer.train_step` 函数则可忽略此条）

例如：

```python
from dataset.dataset import Dataset
from torchvision import datasets, transforms as T
import torch

class MyData(Dataset):
    def __init__(self):
        super().__init__()
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def data_collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return {'image': images, 'label': labels}
```

---

接下来，编写模型类代码，继承自 `models.model.Model` 类，并实现必要的方法：

- `forward`：模型的前向传播函数，接收数据集的输出作为输入，返回一个 `dict`。

此方法必须以前面数据集类的 `data_collate_fn` 方法返回的字典键名作为输入参数，同时返回一个 `dict`，此 `dict` 必须至少包含键 `loss`。（如重写 `Trainer.train_step` 函数则可忽略此条）

```python
from models.model import Model


class MyModel(Model):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, image, label):
        ...
        return {
            'loss': ...,
            'logits': ...,
        }
```

---

编写训练器类代码，继承自 `engine.trainer.Trainer` 类，并实现必要的方法：

- `validate_fn`: 验证函数，以 `dataloader: torch.utils.data.DataLoader` 为参数，返回一个 `dict`，包含各种自定义的测试指标。

例如：

```python
from engine.trainer import Trainer
import torch

class MyTrainer(Trainer):
    def __init__(
            self,
            model,
            dataset,
            cfg,
            exp_dir,
            resume_ckpt=None
    ):
        super().__init__(model, dataset, cfg, exp_dir, resume_ckpt)

    @torch.no_grad()
    def validate_fn(self, dataloader):
        return {
            'val_loss': ...,
            'val_f1': ...,
            'val_recall': ...,
            'val_precision': ...,
            'val_acc': ...
        }
```

部分配置文件内容依赖于此方法返回的字典键名，有：

```yaml
train:
  save:
    save_best_metric: val_f1
  scheduler:
    name: plateau
    params:
      metric: val_acc  # plateau 监控指标，支持 "val_loss", "val_acc" 等
```

`train.save.save_best_metric` 及 `train.scheduler.params.metric` 的值必须包含于 `validate_fn` 方法返回的字典键名中。

可重写 `Trainer.train_step` 函数：

```python
class MyTrainer(Trainer):
    def train_step(self, model, data, optimizer, accelerator):
        """
        单次训练步骤，支持重写
        :param model: 模型实例
        :param data: 输入数据
        :param optimizer: 优化器
        :param accelerator: 加速器实例
        :return: 返回一个包含各种自定义日志记录指标的字典（也可什么都不返回）
        """
        ...
        ### return something as a dict like this, or just return None
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
```

如重写此函数，则可以忽略前面的诸多数据格式要求。

---

编写测试器类代码，继承自 `engine.tester.Tester` 类，并实现必要的方法：

- `test_fn`: 测试函数，以 `dataloader: torch.utils.data.DataLoader` 为参数，返回一个 `dict`，包含各种自定义的测试指标。

```python
from engine.tester import Tester

class MyTester(Tester):
    def __init__(self, model, ckpt_name, dataloader, exp_dir):
        super().__init__(model, ckpt_name, dataloader, exp_dir)

    def test_fn(self, dataloader):
        return {
            'test_loss': ...,
            'test_f1': ...,
            'test_recall': ...,
            'test_precision': ...,
            'test_acc': ...
        }
```

---

接下来，编写脚本，在其中加载配置文件，创建数据集、模型、训练器和测试器，并执行训练和测试流程。

```python
from utils.config import CONFIG
from utils.other_utils import create_exp_dir
from utils.seed import set_seed
from utils.logger import SimpleLogger
import os


if __name__ == "__main__":
    # 创建一个新的实验目录
    exp_dir = create_exp_dir(CONFIG, 'exp')
    # 或者使用指定的已存在的实验目录（从上一个保存断点（或通过 Trainer 的 resume_ckpt 参数指定）继续训练）
    # exp_dir = "runs/exp_20250604_152711"
    # 设置随机种子
    seed = CONFIG.get('seed', 42)
    set_seed(seed)
    # 初始化数据集
    dataset = MyData()
    # 初始化模型
    model = MyModel()
    # 初始化训练器
    trainer = MyTrainer(
        model=model,
        dataset=dataset,
        cfg=CONFIG,
        exp_dir=exp_dir,
        resume_ckpt=None
    )
    # 运行训练器
    trainer.run()
    # 绘制训练过程中的指标图表
    if isinstance(trainer.logger, SimpleLogger):
        trainer.logger.plot(os.path.join(exp_dir, 'plot.png'))
    
    # 初始化测试器
    tester = MyTester(
        model=model,
        ckpt_name='best.pt',
        dataloader=trainer.test_dataloader,
        exp_dir=exp_dir
    )
    # 运行测试器
    tester.run()

```
