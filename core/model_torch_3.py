import json
import math
from typing import *

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from core.layers_torch import CartesianToDihedral, DihedralToCartesian
from core.loader import FragmentDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import matplotlib.pyplot as plt
from matplotlib import colors
import zuko


torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CVAE(pl.LightningModule):
    def __init__(self, n, latent_dim, beta_min, beta_max, lr):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.lr = lr
        self.input_dim = 2*3*n
        self.label_dim = 25*n + 3
        self.latent_dim = latent_dim
        self.c2d = CartesianToDihedral()
        self.d2c = DihedralToCartesian()
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim+self.label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2*self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim+25*14+3,512),
            nn.ReLU(),
            nn.Linear(512, 8192),
            nn.ReLU(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 84),
            # nn.Tanh()
        )
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.prior = zuko.flows.MAF(
            features=self.latent_dim,
            context=25*14+3,
            hidden_features=(512, 512),
            transforms=4,
        )
        self.apply(init_weights)
        self.angles = []
        
    
    def encode(self, x, labels, displacement):
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        x, first_three = self.c2d(x)
        x = torch.cat([x, F.one_hot(aa.type(torch.LongTensor).cuda(), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).cuda(), num_classes=4).flatten(1), displacement], dim=-1)
        x = self.encoder(x)
        mean, log_variance = torch.chunk(x, 2, dim=-1)
        return mean, log_variance, first_three
    
    def sample(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std, device=std.device)
        return mean + std * epsilon
    
    def decode(self, x, labels, displacement, first_three, return_angles=False, train_data=None):
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        x = torch.cat([x.cuda(), F.one_hot(aa.type(torch.LongTensor).cuda(), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).cuda(), num_classes=4).flatten(1), displacement.cuda()], dim=-1)
        x = self.decoder(x) #* torch.full((x.shape[0],42), math.pi, device=self.device)
        x = self.d2c((x, first_three),return_angles=return_angles, train_data=train_data)
        return x
    
    def loss(self, recon_x, recon_x_disp, x, mean, log_variance, displacement, first_three, weights):
        beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (math.cos(math.pi * self.current_epoch/10))
        recon_loss = torch.mean(F.mse_loss(recon_x, x.flatten(1,2)[:,3:,:], reduction='none'))
        kl_loss = torch.mean(-0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()))
        displacement_loss = torch.mean((F.mse_loss(recon_x_disp[:,-1,:] - first_three[:,-1,:], displacement, reduction='none')))
        # displacement_reg = torch.mean((recon_x_disp[:,-1,:] - first_three[:,-1,:]).pow(2))
        loss = recon_loss + beta*kl_loss + .1*displacement_loss #- .001 * displacement_reg
        return loss, recon_loss, kl_loss, displacement_loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        inputs_cart, labels = batch
        weights = labels[:,-1,2].unsqueeze(1).unsqueeze(2)
        labels = labels[:,:,(0,1,3,4,5)]
        
        displacement = inputs_cart[:,-1,-1,:] - inputs_cart[:,0,-1,:]
        
        mean, log_variance, first_three = self.encode(inputs_cart, labels, displacement)
        
        z = self.sample(mean, log_variance)
        recon_inputs = self.decode(z, labels, displacement, first_three, train_data=inputs_cart[:,1:,:,:].flatten(1,2))
        
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        prior = self.prior(torch.cat([F.one_hot(aa.type(torch.LongTensor).cuda(), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).cuda(), num_classes=4).flatten(1), displacement.cuda()], dim=-1))
        prior_loss = -torch.mean(prior.log_prob(z))
        z_1 = prior.sample((1,)).squeeze(0)
        
        recon_inputs_disp = self.decode(z_1, labels, displacement, first_three)
    
        loss, recon_loss, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs_disp, inputs_cart, mean, log_variance, displacement, first_three, weights)
        loss += 0.01 * prior_loss
        
        self.log("train_prior_loss", prior_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_displacement", displacement_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs_cart, labels = batch
        weights = labels[:,-1,2].unsqueeze(1).unsqueeze(2)
        labels = labels[:,:,(0,1,3,4,5)]
        
        displacement = inputs_cart[:,-1,-1,:] - inputs_cart[:,0,-1,:]
        
        mean, log_variance, first_three = self.encode(inputs_cart, labels, displacement)
        z = self.sample(mean, log_variance)
        recon_inputs = self.decode(z, labels, displacement, first_three)
        
        loss, recon_loss, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs, inputs_cart, mean, log_variance, displacement, first_three, weights)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement", displacement_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # benchmark generation
        
        recon_inputs, angles = self.generate(inputs_cart.shape[0], first_three, labels, displacement, return_angles=True)
        loss, recon_loss, kl_loss, displacement = self.loss(recon_inputs, recon_inputs, inputs_cart, mean, log_variance, displacement, first_three, weights)
        below_1 = torch.mean((displacement < 1.0).float())
        self.log("val_recon_loss_generation", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement_generation", displacement, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_below_1_generation", below_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.angles.append(angles)
        
        
    def generate(self, n, first_three, labels, displacement, return_angles=False):
        # z = torch.randn((n, self.latent_dim))
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        prior = self.prior(torch.cat([F.one_hot(aa.type(torch.LongTensor).cuda(), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).cuda(), num_classes=4).flatten(1), displacement.cuda()], dim=-1))
        z = prior.sample((1,)).squeeze(0)
        return self.decode(z, labels, displacement, first_three, return_angles=return_angles)
    
    
    def on_validation_end(self):
        if len(self.angles) == 0:
            return
        angles = torch.cat(self.angles)
        
        # fig, ax = plt.subplots()
        # ax.hist2d(angles[:,0].cpu().detach().numpy(), angles[:,1].cpu().detach().numpy(), bins=[300,300], range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap="Blues", norm=colors.LogNorm())
        # ax.set_xlabel("psi")
        # ax.set_ylabel("omega")
        # ax.set_xlim(-math.pi, math.pi)
        # ax.set_ylim(-math.pi, math.pi)
        # fig.savefig(f'plots/plot_psi_omega_{self.current_epoch}.png')
        # plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.hist2d(angles[:,2].cpu().detach().numpy(), angles[:,0].cpu().detach().numpy(), bins=[300,300], range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap="Blues", norm=colors.LogNorm())
        ax.set_xlabel("phi")
        ax.set_ylabel("psi")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        fig.savefig(f'plots/plot_phi_psi_{self.current_epoch}.png')
        plt.close(fig)
        
        # fig, ax = plt.subplots()
        # ax.hist2d(angles[:,1].cpu().detach().numpy(), angles[:,2].cpu().detach().numpy(), bins=[300,300], range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap="Blues", norm=colors.LogNorm())
        # ax.set_xlabel("omega")
        # ax.set_ylabel("phi")
        # ax.set_xlim(-math.pi, math.pi)
        # ax.set_ylim(-math.pi, math.pi)
        # fig.savefig(f'plots/plot_omega_phi_{self.current_epoch}.png')
        # plt.close(fig)
        
        self.angles = []
    
    
    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()},], lr=self.lr)
    
    def on_train_epoch_end(self):
        torch.save(self.state_dict(), f"models/model_{self.current_epoch}.pt")
        return super().on_train_epoch_end()
    
    
class Trainer():
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config_file = open(config, "r")
        parameters = json.loads(config_file.read())
        
        self.n = parameters["n"]
        self.latent_dim = parameters["latent_dim"]
        self.learning_rate = parameters["learning_rate"]
        self.epochs = parameters["epochs"]
        self.train_batch_size = parameters["train_batch_size"]
        self.val_batch_size = parameters["val_batch_size"]
        self.beta_min = parameters["beta_min"]
        self.beta_max = parameters["beta_max"]
        self.dir_read = parameters["dir_read"]

        config_file.close()
        
        m = FragmentDataModule()
        m.setup(dir_read=self.dir_read)
        self.training_inputs = m.train_dataloader(self.train_batch_size)
        self.test_inputs = m.val_dataloader(self.val_batch_size)   
        
        progress_bar = CustomProgressBar(
            theme=RichProgressBarTheme(
                description="grey82",
                progress_bar="#de7d2f",
                progress_bar_finished="#f5bb47",
                batch_progress="#f5bb47",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
                metrics_text_delimiter="\n",
                metrics_format=".3f",
            )
        )
        
        accelerators = {"cpu": "cpu", "cuda": "gpu"}
        self.trainer = pl.Trainer(accelerator=accelerators[self.device], max_epochs=self.epochs, check_val_every_n_epoch=1, callbacks=[progress_bar], inference_mode=False)

        self.model = CVAE(
            n=self.n,
            latent_dim=self.latent_dim,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            lr=self.learning_rate
        ).to("cuda")
        self.model.to(self.device) 
    
    def train(self):
        self.trainer.fit(self.model, self.training_inputs, self.test_inputs)
        torch.save(self.model.state_dict(), "models/model.pt")
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    def validate(self):
        self.trainer.validate(self.model, self.test_inputs)
        