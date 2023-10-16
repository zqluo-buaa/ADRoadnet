import sys
sys.path.extend([r'D:\dl_project\dl_project_cnn\Utils', r'D:\dl_project\dl_project_cnn\Classfier',
                 r'D:\dl_project\dl_project_cnn\Backbone'])
import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

import sys
sys.path.append(r'D:\dl_project\dl_project_cnn\Utils')
from Utils.modules import normer
from Utils.modules import BasicBlock

from Backbone.FINALNET_FE_HR import FINALNET_FE
from Classfier.FINALNET_CL_HR import training_classifier

from torchstat import stat

class AD_RoadNet(nn.Module):
    def __init__(self, num_classes, norm_layer=None):
        super(AD_RoadNet, self).__init__()
        norm_layer = normer(norm_layer)
        backbone = FINALNET_FE(norm_layer=norm_layer)
        return_layers = {'inconv': 'raw_res', 'rf_attention_pooling_1_1':'1/2_res',
                         'rf_attention_pooling_2_2': '1/4_res', 'rf_attention_pooling_3_4': 'logit'}
        self.backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layers)

        self.classifier = training_classifier(num_classes=num_classes, norm_layer=norm_layer)

        self.out = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
            # norm_layer(32),
            # nn.ReLU(inplace=True),
            # BasicBlock(in_channels=128, out_channels=32, norm_layer=norm_layer),
            # nn.Conv2d(64, num_classes, 3, 1, 1),
            # nn.Conv2d(64, num_classes, 3, 1, 1),
            nn.Conv2d(64, num_classes, 3, 1, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):

        features = self.backbone(x)

        out = self.classifier(features)

        output = self.out(out)

        return output


if __name__ == '__main__':
    net = FirstStagenet_3_3(num_classes=1, norm_layer='GN')

    stat(net, (3, 512, 512))

