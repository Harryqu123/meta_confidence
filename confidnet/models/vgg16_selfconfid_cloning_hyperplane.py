from confidnet.models.model import AbstractModel
from confidnet.models.vgg16 import VGG16
from confidnet.models.vgg16_selfconfid_classic import VGG16SelfConfidClassic
from confidnet.models.vgg16_selfconfid_classic_hyperplane_5 import VGG16SelfConfidClassicHyperplane5


class VGG16SelfConfidCloningHyperplane(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.pred_network = VGG16(config_args, device)

        # Small trick to set num classes to 1
        temp = config_args["data"]["num_classes"]
        config_args["data"]["num_classes"] = 1
        self.uncertainty_network = VGG16SelfConfidClassicHyperplane5(config_args, device)
        config_args["data"]["num_classes"] = temp

    def forward(self, x):
        pred = self.pred_network(x)
        _, uncertainty = self.uncertainty_network(x)
        return pred, uncertainty
