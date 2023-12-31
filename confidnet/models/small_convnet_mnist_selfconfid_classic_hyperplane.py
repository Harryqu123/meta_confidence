import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel

import torch


class SmallConvNetMNISTSelfConfidClassicHyperplane(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.conv1 = nn.Conv2d(config_args["data"]["input_channels"], 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, config_args["data"]["num_classes"])

        self.temperature = 1e-7
        self.uncertainty_linear = nn.Linear(128, 1, bias=True)

        # self.uncertainty1 = nn.Linear(128, 400)
        # self.uncertainty2 = nn.Linear(400, 400)
        # self.uncertainty3 = nn.Linear(400, 400)
        # self.uncertainty4 = nn.Linear(400, 400)
        # self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.25, training=self.training)
        else:
            out = self.dropout1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(out, 0.5, training=self.training)
        else:
            out = self.dropout2(out)

        uncertainty = self.uncertainty_linear(out) / (torch.pow(self.uncertainty_linear.weight, 2).sum().sqrt() + self.temperature)


        # uncertainty = F.relu(self.uncertainty1(out))
        # uncertainty = F.relu(self.uncertainty2(uncertainty))
        # uncertainty = F.relu(self.uncertainty3(uncertainty))
        # uncertainty = F.relu(self.uncertainty4(uncertainty))
        # uncertainty = self.uncertainty5(uncertainty)
        pred = self.fc2(out)
        return pred, uncertainty
