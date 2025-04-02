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
from typing import Tuple

import numpy as np


torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPS = 1e-8


class CustomProgressBar(RichProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CVAE(pl.LightningModule):
    def __init__(self, n, latent_spec, beta_min, beta_max, lr):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.lr = lr
        self.input_dim = 2 * 3 * n
        self.label_dim = 25 * n + 3
        self.latent_spec = latent_spec
        self.c2d = CartesianToDihedral()
        self.d2c = DihedralToCartesian()

        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.num_pixels = self.input_dim + self.label_dim
        self.temperature = 0.67
        self.hidden_dim = 256

        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim+self.label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 8192),
            nn.ReLU(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 84),
        )

        if self.is_continuous:
            self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

        self.angles = []

    def encode(self, x, labels, displacement):
        aa, ss = labels[:, 1:, 0].long(), labels[:, 1:, 1].long()
        x, first_three = self.c2d(x)
        x = torch.cat([x, F.one_hot(aa.type(torch.LongTensor).to(self.device), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).to(self.device), num_classes=4).flatten(1), displacement], dim=-1)
        batch_size = x.size()[0]

        hidden = self.encoder(x)

        latent_dist = {}

        if self.is_continuous:
            mean = self.fc_mean(hidden)
            logvar = self.fc_log_var(hidden)
            latent_dist['cont'] = [mean, logvar]

        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

        return latent_dist, first_three

    def reparameterize(self, latent_dist):
        latent_sample = []

        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            eps = eps.to(self.device)
            return mean + std * eps
        else:
            return mean

    def sample_gumbel_softmax(self, alpha):
        if self.training:
            unif = torch.rand(alpha.size()).to(self.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            one_hot_samples = one_hot_samples.to(self.device)
            return one_hot_samples

    def decode(self, latent_sample, labels, displacement, first_three, return_angles=False, train_data=None):
        aa, ss = labels[:, 1:, 0].long(), labels[:, 1:, 1].long()
        x = torch.cat([latent_sample.to(self.device), F.one_hot(aa.type(torch.LongTensor).to(self.device), num_classes=21).flatten(1), F.one_hot(ss.type(torch.LongTensor).to(self.device), num_classes=4).flatten(1), displacement.to(self.device)], dim=-1)
        x = self.decoder(x)
        x = self.d2c((x, first_three), return_angles=return_angles, train_data=train_data)
        return x

    def loss(self, recon_x, recon_x_disp, x, latent_dist, displacement, first_three, weights):
        beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (math.cos(math.pi * self.current_epoch / 10))
        recon_loss = torch.mean(F.mse_loss(recon_x, x.flatten(1, 2)[:, 3:, :], reduction='none'))
        kl_loss = self._kl_normal_loss(latent_dist['cont'][0], latent_dist['cont'][1]) + self._kl_multiple_discrete_loss(latent_dist['disc'])
        displacement_loss = torch.mean(F.mse_loss(recon_x_disp[:, -1, :] - first_three[:, -1, :], displacement, reduction='none'))
        loss = recon_loss + beta * kl_loss + 0.1 * displacement_loss
        return loss, recon_loss, kl_loss, displacement_loss


    def _kl_normal_loss(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        kl_loss = torch.sum(kl_means)
        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]
        kl_loss = torch.sum(torch.cat(kl_losses))
        return kl_loss

    def _kl_discrete_loss(self, alpha):
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        log_dim = log_dim.to(self.device)
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        inputs_cart, labels = batch
        weights = labels[:, -1, 2].unsqueeze(1).unsqueeze(2)
        labels = labels[:, :, (0, 1, 3, 4, 5)]

        displacement = inputs_cart[:, -1, -1, :] - inputs_cart[:, 0, -1, :]

        latent_dist, first_three = self.encode(inputs_cart, labels, displacement)

        z = self.reparameterize(latent_dist)
        recon_inputs = self.decode(z, labels, displacement, first_three, train_data=inputs_cart[:, 1:, :, :].flatten(1, 2))
        recon_inputs_disp = self.decode(z, labels, displacement, first_three)

        loss, recon_loss, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs_disp, inputs_cart, latent_dist, displacement, first_three, weights)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_displacement", displacement_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs_cart, labels = batch
        weights = labels[:, -1, 2].unsqueeze(1).unsqueeze(2)
        labels = labels[:, :, (0, 1, 3, 4, 5)]

        displacement = inputs_cart[:, -1, -1, :] - inputs_cart[:, 0, -1, :]

        latent_dist, first_three = self.encode(inputs_cart, labels, displacement)
        z = self.reparameterize(latent_dist)
        recon_inputs = self.decode(z, labels, displacement, first_three)

        loss, recon_loss, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs, inputs_cart, latent_dist, displacement, first_three, weights)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement", displacement_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        recon_inputs, angles = self.generate(inputs_cart.shape[0], first_three, labels, displacement, return_angles=True)
        loss, recon_loss, kl_loss, displacement = self.loss(recon_inputs, recon_inputs, inputs_cart, latent_dist, displacement, first_three, weights)
        below_1 = torch.mean((displacement < 1.0).float())
        self.log("val_recon_loss_generation", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement_generation", displacement, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_below_1_generation", below_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.angles.append(angles)

    def generate(self, n, first_three, labels, displacement, return_angles=False):
        z = torch.randn((n, self.latent_dim))
        return self.decode(z, labels, displacement, first_three, return_angles=return_angles)

    def on_validation_end(self):
        if len(self.angles) == 0:
            return
        angles = torch.cat(self.angles)

        fig, ax = plt.subplots()
        ax.hist2d(angles[:, 2].cpu().detach().numpy(), angles[:, 0].cpu().detach().numpy(), bins=[300, 300], range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap="Blues", norm=colors.LogNorm())
        ax.set_xlabel("phi")
        ax.set_ylabel("psi")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        fig.savefig(f'plots/plot_phi_psi_{self.current_epoch}.png')
        plt.close(fig)

        self.angles = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
            latent_spec={"cont":self.latent_dim, "disc":[1]*28},
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            lr=self.learning_rate
        ).to(self.device)
        self.model.to(self.device)

    def train(self):
        self.trainer.fit(self.model, self.training_inputs, self.test_inputs)
        torch.save(self.model.state_dict(), "models/model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def validate(self):
        self.trainer.validate(self.model, self.test_inputs)
