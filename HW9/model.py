import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2), # [32, 16, 16]

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2), # [64, 8, 8]

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2), # [128, 4, 4]

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2)  # [256, 2, 2]                       
            
        )
 
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(256, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(128, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(64, 32, 9, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(32, 3, 17, stride=1),
            nn.Tanh()
        )
        self.latent_size = 512
        self.fc1 = nn.Linear(1024, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, 1024)

    def forward(self, x):
        x1 = self.encoder(x)
        latent = self.fc1(x1.reshape(-1, 1024))
        x2 = self.fc2(latent)
        x2 = x2.view(-1, 256, 2, 2)
        x  = self.decoder(x2)
        return latent, x

class baseline_AE(nn.Module):
    def __init__(self):
        super(baseline_AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x