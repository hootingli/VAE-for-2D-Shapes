import torch
import torch.nn.functional as F
from torch import nn


class ResLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, repeat=0, bias=True):
        super(ResLinearBlock, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_features)
            nn.LayerNorm(hidden_features)
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features, bias=bias),
                nn.ReLU(),
                # nn.BatchNorm1d(hidden_features)
                nn.LayerNorm(hidden_features)
                )
                for _ in range(repeat)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features, bias=bias),
            nn.ReLU(),
            # nn.BatchNorm1d(out_features)
            nn.LayerNorm(out_features)
        )
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

    def forward(self, x):
        x1 = self.input_layer(x)
        for layer in self.hidden_layers:
            x1 = layer(x1)
        x1 = self.output_layer(x1)
        return x1 + x[:, :self.out_features]
    
class ResVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, h_layers=[2,2], z_dim=2):
        super().__init__()
        # encoder
        self.img2hid = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim)
            # nn.LayerNorm(h_dim)
        )
        self.encoder_layers = nn.ModuleList([
            ResLinearBlock(h_dim, h_dim, h_dim, repeat=repeat)
            for repeat in h_layers
        ])
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2log_var = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hid = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim)
            # nn.LayerNorm(h_dim)
        )
        self.decoder_layers = nn.ModuleList([
            ResLinearBlock(h_dim, h_dim, h_dim, repeat=repeat)
            for repeat in h_layers
        ])
        self.hid2img = nn.Sequential(
            nn.Linear(h_dim, input_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(input_dim)
            # nn.LayerNorm(input_dim)
        )

        # Define the learnable alpha parameter
        self.mean = nn.Parameter(torch.tensor(3.168751), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(1.014383), requires_grad=True)

    def encode(self, x):
        h = self.img2hid(x)
        for layer in self.encoder_layers:
            h = layer(h)
        mu, log_var = self.hid2mu(h), self.hid2log_var(h)
        return mu, log_var

    def decode(self, z, eps=1e-8):
        #p_theta(x|z)
        h = self.z2hid(z)
        for layer in self.decoder_layers:
            h = layer(h)
        h = self.hid2img(h)
        # h = (h - h.mean()) / h.std()
        # h = h * self.std + self.mean
        return h
        #return torch.tanh(self.hid_2img(h))
        #return self.hid_2img(h)  # No activation

    def forward(self, x):
        mu, log_var = self.encode(x)
        x_reconstructed = self.generate(mu, log_var)
        return x_reconstructed, mu, log_var
    
    def generate(self, mu, log_var):
        sigma = log_var.exp().sqrt()
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed
    
    def reconstruct(self, h, eps=1e-8):
        h = 1 / (torch.exp(h)+eps)
        h = h / torch.sum(h, dim=1, keepdim=True) * 2 * torch.pi
        return h
    
    def loss(self, x, x_reconstructed, mu, log_var, kl_rate=0.01):
        # Reconstruction loss
        # recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        recon_loss = (x_reconstructed - x).pow(2).sum(dim=1).mean()
        true_loss = (self.reconstruct(x) - self.reconstruct(x_reconstructed)).pow(2).sum(dim=1).mean()

        # KL divergence loss
        kl_loss = mu.pow(2) + log_var.exp() - log_var - 1
        kl_loss = 0.5 * torch.sum(kl_loss, dim=1).mean()
        loss = recon_loss + kl_rate * kl_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'true_loss': true_loss
        }
    
    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass