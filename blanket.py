#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F


class Blanket(torch.autograd.Function):
    @staticmethod
    def normalize(x):
        y = x.flatten(1)
        y /= y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6
        y *= math.sqrt(y.numel() / y.size(0))

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        # Normalize the forward
        Blanket.normalize(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clone()
        # Normalize the gradient
        Blanket.normalize(grad_output)
        return grad_output


blanket = Blanket.apply

######################################################################

if __name__ == "__main__":
    x = torch.rand(2, 3).requires_grad_()
    y = blanket(x) * 10
    print(y.pow(2).sum())
    z = y.sin().sum()
    g = torch.autograd.grad(z, x)[0]

    print(g.pow(2).sum())
