import torch
import torch.nn.functional as F
from torch import nn

class DeepVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=500, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid1 = nn.Linear(input_dim, h_dim)
        self.hid1_2hid2 = nn.Linear(h_dim, h_dim)  # New hidden layer
        self.hid2_2mu = nn.Linear(h_dim, z_dim)
        self.hid2_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid1 = nn.Linear(z_dim, h_dim)
        self.hid1_2hid2 = nn.Linear(h_dim, h_dim)  # New hidden layer
        self.hid2_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

        # Define the learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.75), requires_grad=True)

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid1(x))
        h = self.relu(self.hid1_2hid2(h))  # New hidden layer
        mu, sigma = self.hid2_2mu(h), self.hid2_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid1(z))
        h = self.relu(self.hid1_2hid2(h))  # New hidden layer
        return torch.tanh(self.hid2_2img(h))

    def forward(self, x):
        self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)  # 28x28 = 784
    vae = DeepVariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed)
    print(mu)
    print(sigma)
    print(vae(x).shape)
