import torch
import torch.nn.functional as F
from torch import nn

class SoftIntroVAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=2):
        super(SoftIntroVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        # Decoder
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == "__main__":
    x = torch.randn(4, 200) # Adjusted to match 200x1 input size
    model = SoftIntroVAE(input_dim=200, h_dim=200, z_dim=2)
    reconstructed_x, mu, logvar = model(x)
    print(reconstructed_x.shape, mu.shape, logvar.shape)
