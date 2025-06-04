from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from itertools import cycle


def denorm(img: torch.Tensor, mean, std):
    if isinstance(mean, float):
        mean = (mean,)
    if isinstance(std, float):
        std = (std,)
    img = img.clone()
    for t, m, s in zip(img, cycle(mean), cycle(std)):
        t.mul_(s).add_(m)
    return img


def imshow(img: torch.Tensor, save_path: str = None):
    img = img.numpy()
    fig = plt.figure(figsize=(img.shape[2] / 100, img.shape[1] / 100), dpi=100)
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def multi_imshow(imgs: torch.Tensor, nrow=None, save_path: str = None, **kwargs):
    if nrow is None:
        nrow = int(np.sqrt(imgs.shape[0]))
    imshow(torchvision.utils.make_grid(imgs, nrow=nrow, **kwargs), save_path)