import torch
import torch.nn.functional as F
from torch import nn


class ResLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, repeat=0, bias=True):
        super(ResLinearBlock, self).__init__()
        self.input_layer = nn.Linear(in_features, hidden_features, bias=bias)
        self.hidden_layers = nn.ModuleList([
                nn.Linear(hidden_features, hidden_features, bias=bias) 
                for _ in range(repeat)])
        self.output_layer = nn.Linear(hidden_features, out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

    def forward(self, x):
        x1 = self.input_layer(x)
        x1 = torch.relu(x1)
        for layer in self.hidden_layers:
            x1 = layer(x1)
            x1 = torch.relu(x1)
        x1 = self.output_layer(x1)
        return torch.relu(x1 + x[:, :self.out_features])
    
class ResVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, h_layers=[2,2], z_dim=2):
        super().__init__()
        # encoder
        self.img2hid = nn.Linear(input_dim, h_dim)
        self.encoder_layers = nn.ModuleList([
            ResLinearBlock(h_dim, h_dim, h_dim, repeat=repeat)
            for repeat in h_layers
        ])
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hid = nn.Linear(z_dim, h_dim)
        self.decoder_layers = nn.ModuleList([
            ResLinearBlock(h_dim, h_dim, h_dim, repeat=repeat)
            for repeat in h_layers
        ])
        self.hid2img = nn.Linear(h_dim, input_dim)

        # Define the learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.75), requires_grad=True)

    def encode(self, x):
        h = torch.relu(self.img2hid(x))
        for layer in self.encoder_layers:
            h = layer(h)
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    def decode(self, z):
        #p_theta(x|z)
        h = torch.relu(self.z2hid(z))
        for layer in self.decoder_layers:
            h = layer(h)
        h = self.hid2img(h)
        h = torch.sigmoid(h)
        # normalize the output
        h = h / torch.sum(h, dim=1, keepdim=True) * 2 * torch.pi
        return h
        #return torch.tanh(self.hid_2img(h))
        #return self.hid_2img(h)  # No activation

    def forward(self, x):
        self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        mu, sigma = self.encode(x)
        x_reconstructed = self.generate(mu, sigma)
        return x_reconstructed, mu, sigma
    
    def generate(self, mu, sigma):
        var = torch.exp(sigma)
        epsilon = torch.randn_like(var)
        z_reparametrized = mu + var * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed
    
    def loss(self, x, x_reconstructed, mu, sigma, kl_rate=0.01):
        # Reconstruction loss
        # recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        recon_loss = (x_reconstructed - x).pow(2).sum(dim=1).mean()

        # KL divergence loss
        kl_loss = mu.pow(2) + sigma.exp() - sigma - 1
        kl_loss = 0.5 * torch.sum(kl_loss, dim=1).mean()
        loss = recon_loss + kl_rate * kl_loss
        return loss, recon_loss, kl_loss
    
    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)