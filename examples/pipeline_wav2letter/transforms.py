import torch


class Normalize(torch.nn.Module):
    def forward(self, tensor):
        return (tensor - tensor.mean(-1, keepdim=True)) / tensor.std(-1, keepdim=True)


class UnsqueezeFirst(torch.nn.Module):
    def forward(self, tensor):
        return tensor.unsqueeze(0)


class ToMono(torch.nn.Module):
    def forward(self, tensor):
        return tensor[0, ...]
