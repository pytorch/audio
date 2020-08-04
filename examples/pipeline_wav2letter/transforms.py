import torch


class Normalize(torch.nn.Module):
    def forward(self, tensor):
        return (tensor - tensor.mean(-1, keepdim=True)) / tensor.std(-1, keepdim=True)
