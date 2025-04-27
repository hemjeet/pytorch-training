import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, input_size, class_idx):
        super().__init__()
        self.input_size = input_size
        self.num_classes = len(class_idx)
        
        if len(class_idx) == 2:
            self.output = self.num_classes - 1
        else:
            self.output = self.num_classes
        
        #--------------convolutional layer----------------#
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_size[1], 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  
        )

        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size[1:])
            n_features = self.flatten(self.conv(dummy)).shape[1]

        #-------------fully connected layer------------#
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.output)
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
