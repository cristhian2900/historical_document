import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 64 * 64),  
            nn.Tanh()  
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), -1, 64, 64) 
        return out