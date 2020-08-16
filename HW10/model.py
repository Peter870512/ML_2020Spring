import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3)
            )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 32 * 32 * 3), 
            nn.Tanh()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(input, output, kernel, stride, padding)
            nn.Conv2d(3, 32, 3, stride=1, padding=1),            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                               # [batch, 32, 16, 16]

            nn.Conv2d(32, 64, 3, stride=1, padding=1),           
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                               # [batch, 64, 8, 8] 

			nn.Conv2d(64, 128, 3, stride=1, padding=1),          
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),                               # [batch, 128, 4, 4]

            nn.Conv2d(128, 256, 3, stride=1, padding=1),          
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)                               # [batch, 256, 2, 2]

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1),  # [batch, 128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

			nn.ConvTranspose2d(128, 64, 5, stride=1),  # [batch, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),

			nn.ConvTranspose2d(64, 32, 9, stride=1),  # [batch, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 17, stride=1),   # [batch, 3, 32, 32]
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

        self.fc1 = nn.Linear(256 * 2 * 2, 200)
        #self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(100, 256 * 2 * 2)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        h1 = self.fc1(x)
        return h1

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = self.fc4(z)
        h4 = h4.view(h4.size(0), 256, 2, 2)
        out = self.decoder(h4)
        return out

    def forward(self, x):
        vector = self.encode(x)
        mu = vector[:, :100]
        logvar = vector[:, 100:]
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z
        #return self.decode(x)


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return mse + KLD