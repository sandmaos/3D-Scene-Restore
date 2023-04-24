import copy

import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module
from mmflow.apis import init_model, inference_model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, same, strid=2, kernel_size=3):
        super().__init__()
        # The parameter same decides whether we should reduce the output size
        # if same is true , the res connection would add the original input
        # else it would add a down sampled input.

        if not same:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strid)
            self.res_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strid),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.res_connection = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        res_connection = self.res_connection(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + res_connection
        return nn.ELU()(input)

#The class Net defines the structure of the nerual network
class Net(nn.Module):
    def __init__(self, resblock):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.a1=torch.ones((1, 1, 378 ,504)).to(device)
        self.a2=torch.ones((1, 1, 378, 504)).to(device)
        self.a3=torch.ones((1, 1, 378, 504)).to(device)
        self.conv_x=nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv_y=nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.conv_z=nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.layer0 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, True),
            resblock(64, 64, True)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, False ,3 , (3,3)),
            resblock(128, 128, True)
        )


        self.layer3 = nn.Sequential(
            resblock(128, 256, False, 2, (3, 3)),
            resblock(256, 256, True)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 256, False, 3, (3,3)),
            resblock(256, 256, True)
        )

        self.layer5 = nn.Sequential(
            resblock(256, 512, False, 2, (3, 3)),
            resblock(512, 512, True)
        )
        self.layer6=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(5,5)),
            nn.Conv2d(512, 512, kernel_size=(5,5)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 3),
        )
        self.layer0_t = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.layer1_t = nn.Sequential(
            resblock(64, 64, True),
            resblock(64, 64, True)
        )

        self.layer2_t = nn.Sequential(
            resblock(64, 128, False, 3, (3, 3)),
            resblock(128, 128, True)
        )

        self.layer3_t = nn.Sequential(
            resblock(128, 256, False, 2, (3, 3)),
            resblock(256, 256, True)
        )

        self.layer4_t = nn.Sequential(
            resblock(256, 256, False, 3, (3, 3)),
            resblock(256, 256, True)
        )

        self.layer5_t = nn.Sequential(
            resblock(256, 512, False, 2, (3, 3)),
            resblock(512, 512, True)
        )
        self.layer6_t_ = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(5, 5)),
            nn.Conv2d(512, 512, kernel_size=(5, 5)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 9),

        )

    def forward(self, input):

        pred_translation = self.layer0(input)
        pred_translation = self.layer1(pred_translation)
        pred_translation = self.layer2(pred_translation)
        pred_translation = self.layer3(pred_translation)
        pred_translation = self.layer4(pred_translation)
        pred_translation = self.layer5(pred_translation)
        pred_translation = self.layer6(pred_translation)/10

        pred_translation= nn.Tanh()(pred_translation)

        x=torch.mul(self.a1, pred_translation[0,0])
        y=torch.mul(self.a2, pred_translation[0,1])
        z=torch.mul(self.a3, pred_translation[0,2])
        x=self.conv_x(x)
        y=self.conv_y(y)
        z=self.conv_z(z)

        pred_translation = pred_translation.reshape(3)

        new_input=torch.clone(input)
        new_input[0, 0] = new_input[0, 0] - x
        new_input[0, 1] = new_input[0, 1] - y

        new_input=new_input * z

        pred_rotation = self.layer0_t(new_input)
        pred_rotation = self.layer1_t(pred_rotation)
        pred_rotation = self.layer2_t(pred_rotation)
        pred_rotation = self.layer3_t(pred_rotation)
        pred_rotation = self.layer4_t(pred_rotation)
        pred_rotation = self.layer5_t(pred_rotation)
        pred_rotation = self.layer6_t_(pred_rotation) / 10
        pred_rotation = nn.Tanh()(pred_rotation)
        pred_rotation =pred_rotation.reshape(3,3)

        return  pred_translation,pred_rotation
