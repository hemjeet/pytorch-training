import torch.nn as nn
import torch

class ModelV(nn.Module):
    def __init__(
            self, 
            input_size, 
            class_idx, 
            pool,
            conv_chanels = None,
            fc_channels = None,
            kernel_size = 3, 
            padding = 1, 
            pool_size = 2

    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = len(class_idx)
        self.conv_chanels = conv_chanels
        self.fc_channels = fc_channels
        self.kernel_size = kernel_size or 3
        self.padding = padding or 1
        self.pool_size = pool_size or 2
        self.pool = pool

       
        if len(class_idx) == 2:
            self.output = self.num_classes - 1
        else:
            self.output = self.num_classes
        
        if self.conv_chanels is None:
            raise ValueError('Conv channels cant be None !')
        
        #------------// feature extraction layer //------------#
        self.conv_layers = self._make_conv_layers()
        
        #----------------// avg pooling //-----------------------#
        if self.pool == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        #------------// flattened layer //----------------#
        self.flatten = nn.Flatten()

        #--------------fully connected layer--------------#
        self.classifier = self._make_fc_layers()

    def get_fc_input(self):
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size[1:])
            if self.pool == 'avg':
                n_features = self.flatten(self.avg_pool(self.conv_layers(dummy))).shape[1]
            else:
                n_features = self.flatten(self.conv_layers(dummy)).shape[1]

        return n_features
        
    def _make_conv_layers(self):
        layers = []
        in_channels = self.input_size[1]
        for out_channels in self.conv_chanels:
            seq_layers = [
                nn.Conv2d(in_channels, out_channels, self.kernel_size, self.padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            if self.pool == 'max':
                seq_layers.append(nn.MaxPool2d(self.pool_size))
            layers.extend(seq_layers)
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_fc_layers(self):
        fc_layers = list()
        in_features = self.get_fc_input()
        for out_features in self.fc_channels:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU(inplace = True))  # inplace saves memory
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, self.output))
        return nn.Sequential(*fc_layers)

    
    def forward(self, x):
        x = self.conv_layers(x)
        if self.pool == 'avg':
            x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

