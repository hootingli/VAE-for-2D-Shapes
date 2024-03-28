import torch
from torch import nn



# Input image -> Hidden dim -> mean, std -> Parametrization Trick _> Decoder -> Output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=2):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2hid1 = nn.Linear(h_dim, h_dim)
        self.hid_2hid2 = nn.Linear(h_dim, h_dim)
        self.hid_2hid3 = nn.Linear(h_dim, h_dim)
        self.hid_2hid4 = nn.Linear(h_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2hid5 = nn.Linear(h_dim, h_dim)
        self.hid_2hid6 = nn.Linear(h_dim, h_dim)
        self.hid_2hid7 = nn.Linear(h_dim, h_dim)
        self.hid_2hid8 = nn.Linear(h_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

        # Define the learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.75), requires_grad=True)
        


    
    def encode(self, x):
        #q_phi(z|x)
        h = self.img_2hid(x)
        h = self.hid_2hid1(h)
        #h = self.hid_2hid2(h)
        #h = self.hid_2hid3(h)
        #h = self.hid_2hid4(h)
        h = self.relu(h)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
        

    def decode(self, z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        h = self.hid_2hid5(h)
        #h = self.hid_2hid6(h)
        #h = self.hid_2hid7(h)
        #h = self.hid_2hid8(h)
        h =  self.hid_2img(h)
        return torch.sigmoid(h)
        #return torch.tanh(self.hid_2img(h))
        #return self.hid_2img(h)  # No activation

    def forward(self, x):
        self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    


if __name__ == "__main__":
    x = torch.randn(4, 28*28) #28x28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed)
    print(mu)
    print(sigma)
    # print(vae(x).shape)