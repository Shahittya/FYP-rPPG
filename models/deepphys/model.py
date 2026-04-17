import torch
import numpy as np
import torch.nn as nn
class DeepPhysModel(nn.Module):
    def __init__(self):
        super(DeepPhysModel, self).__init__()
        #Motion
        self.motion_stream =nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        #Appearance
        self.appearance_stream = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        #Fully connected layers
        self.fc=nn.Sequential(
            nn.Linear(32*72*72,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,appearance,motion):
            motion=motion.permute(0,3,1,2) #B,C,H,W
            appearance=appearance.permute(0,3,1,2)
            m=self.motion_stream(motion)
            a=self.appearance_stream(appearance)
            x=m*a
            x=x.reshape(x.size(0),-1)
            x=self.fc(x)
            return x.squeeze()