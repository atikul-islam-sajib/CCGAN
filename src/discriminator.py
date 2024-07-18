import sys
import torch
import torch.nn as nn

sys.path.append("./")


class Discriminator(nn.Module):
    def __init__(self, input_size: tuple = (1, 3, 128, 128)):
        super(Discriminator, self).__init__()

        self.batch_size, self.channels, self.image_height, self.image_width = input_size

        self.layers = []

        pass
