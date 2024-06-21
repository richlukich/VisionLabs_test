import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

class VanillaVAE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims[:3]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Sequential(
                       nn.Conv2d(in_channels, out_channels=hidden_dims[-1],
                              kernel_size= 3, stride= 3, padding  = 0),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU()) 
                    )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        modules.append(nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[0],
                                       hidden_dims[1],
                                       kernel_size=3,
                                       stride = 3,
                                       padding=0),
                    nn.BatchNorm2d(hidden_dims[1]),
                    nn.LeakyReLU()))
        
        for i in range(1, len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(32,
                                       1,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                            nn.Tanh())
    
    def encode(self, input: torch.Tensor) -> list[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        result = self.decoder_input(z)
        result = result.view(-1, 256, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]
    

class EyeClassifier(nn.Module):
    def __init__(self, latent_dim, pretrained_vae=None):
        super().__init__()
        '''
        self.vae = VAE(latent_dim)
        try:
            self.vae.load_state_dict(torch.load(pretrained_vae))

            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

            self.class_fc = nn.Linear(latent_dim, 1)
        except FileNotFoundError as e:
            print(e)
            print('Укажите путь к весам VAE!')
        '''
        self.encoder = Encoder(latent_dim)
        self.freeze_backbone = pretrained_vae is not None
        if pretrained_vae is not None:
            try:
                state_dict = torch.load(pretrained_vae)
                state_dict = OrderedDict(((k[len("encoder."):], v)
                                          for k, v in state_dict.items()
                                          if "encoder." in k))  
                self.encoder.load_state_dict(state_dict, strict=True)
                for param in self.encoder.parameters():
                    param.requires_grad = False
            except FileNotFoundError as e:
                print(e)
                print('Укажите путь к весам VAE!')
        self.encoder.eval()
        self.class_fc = nn.Linear(latent_dim, 1)    

    
    def forward(self, x):
        with torch.no_grad():
            mu, log_var = self.encoder(x)

        logits = self.class_fc(mu)
        x = torch.sigmoid(logits)
        x = x.squeeze(-1)
        return x
    def train(self, mode=True):
        
        self.encoder.train(False)
        self.class_fc.train(mode)



class Encoder(nn.Module):

    """ Class for VAE encoder"""

    def __init__(self, latent_size):
        """
        :param latent_size: int
            Dimension of the latent space
        """
        super().__init__()

        self.conv1 = BatchNorm(nn.Conv2d(1, 32, 3, padding=1, stride=2))
        self.conv2 = BatchNorm(nn.Conv2d(32, 64, 3, padding=1, stride=2))
        self.conv3 = BatchNorm(nn.Conv2d(64, 128, 3, padding=1, stride=2))
        self.conv4 = BatchNorm(nn.Conv2d(128, 256, 3, padding=0, stride=3))

        self.mu_fc = nn.Linear(256, latent_size)
        self.log_var_fc = nn.Linear(256, latent_size)

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


class BatchNorm(nn.Module):

    

    def __init__(self, op):
        super().__init__()
        self.op = op
        self.bn = nn.BatchNorm2d(op.out_channels)

    def forward(self, x):
        x = self.op(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):

    

    def __init__(self, latent_size):
        
        super().__init__()

        self.dec_fc = nn.Linear(latent_size, 256)

        self.tconv1 = nn.ConvTranspose2d(256, 128, 3, padding=0, stride=3)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 3, output_padding=0, stride=2)
        self.tconv3 = nn.ConvTranspose2d(64, 32, 3, output_padding=0, stride=2)
        self.tconv4 = nn.ConvTranspose2d(32, 1, 3, output_padding=0, stride=2)

    def forward(self, z):
        
        x = self.dec_fc(z)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.tconv1(x))
        x = torch.relu(self.tconv2(x))[:, :, :-1, :-1]
        x = torch.relu(self.tconv3(x))[:, :, :-1, :-1]
        x = torch.sigmoid(self.tconv4(x))[:, :, :-1, :-1]

        return x


class VAE(nn.Module):

    

    def __init__(self, latent_size):
        
        super().__init__()

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
       
        mu, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x = self.decoder(z)

        return x, z, mu, std