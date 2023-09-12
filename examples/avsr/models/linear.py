import torch.nn as nn


class VisualFrontendLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(VisualFrontendLinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size, time, _, height, width = x.size()
        x = x.view(batch_size, time, height * width)
        x = self.linear(x)
        return x

def video_linear():
    return VisualFrontendLinearLayer(1936, 512)
