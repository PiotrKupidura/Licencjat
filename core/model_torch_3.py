import json
import math
from typing import *

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from transformers import BertConfig, BertModel

from core.layers_torch import CartesianToDihedral, DihedralToCartesian
from core.loader import FragmentDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import matplotlib.pyplot as plt
from matplotlib import colors

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CustomProgressBar(RichProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, attention_dim):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, attention_dim)
        self.key_projection = nn.Linear(key_dim, attention_dim)
        self.value_projection = nn.Linear(value_dim, attention_dim)
        self.scale = attention_dim ** -0.5

    def forward(self, query, key, value):
        Q = self.query_projection(query)
        K = self.key_projection(key)
        V = self.value_projection(value)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_value = torch.matmul(attention_weights, V)
        return attended_value, attention_weights

class OrientationEmbedding(nn.Module):
    def __init__(self, orientation_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Linear(orientation_dim, embedding_dim)

    def forward(self, orientation):
        return self.embedding(orientation)

class InputAttention(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dim, orientation_dim, embedding_dim, attention_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        ) # Process each element independently
        self.orientation_embedding = OrientationEmbedding(orientation_dim, embedding_dim)
        self.attention = Attention(mlp_hidden_dim, embedding_dim, embedding_dim, attention_dim)
        self.output_layer = nn.Linear(mlp_hidden_dim + embedding_dim, output_dim)

    def forward(self, input_sequence, orientation):
        # input_sequence: (batch_size, seq_len, input_dim)
        # orientation: (batch_size, orientation_dim)

        embedded_orientation = self.orientation_embedding(orientation).unsqueeze(1) # (batch_size, 1, embedding_dim)

        mlp_output = self.mlp(input_sequence) # (batch_size, seq_len, mlp_hidden_dim)

        attended_value, attention_weights = self.attention(mlp_output, embedded_orientation, embedded_orientation) # (batch_size, seq_len, embedding_dim)

        concatenated_input = torch.cat((mlp_output, attended_value), dim=-1) # (batch_size, seq_len, mlp_hidden_dim + embedding_dim)

        output = self.output_layer(concatenated_input).squeeze(1) # (batch_size, seq_len, output_dim)
        return output, attention_weights

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(condition_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.beta = nn.Sequential(
            nn.Linear(condition_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, condition):
        x = self.linear(x)
        x = self.bn(x)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        x = gamma * x + beta
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CVAE(pl.LightningModule):
    def __init__(self, n, latent_dim, beta_min, beta_max, lr):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.lr = lr
        self.input_dim = 3*n
        self.label_dim = 25*n + 3
        self.latent_dim = latent_dim
        self.c2d = CartesianToDihedral()
        self.d2c = DihedralToCartesian()
        self.d_model = 32
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)

        # Initialize BERT model with random weights
        config = BertConfig(hidden_size=self.d_model, num_hidden_layers=3, num_attention_heads=4, intermediate_size=1024,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.encoder = BertModel(config)
        self.decoder = BertModel(config)

        self.input_mu = nn.Linear(3, self.d_model)
        self.input_sigma = nn.Linear(3, self.d_model)
        self.output_mu = nn.Linear(self.d_model, self.latent_dim)
        self.output_sigma = nn.Linear(self.d_model, self.latent_dim)
        self.decoder_projection = nn.Linear(self.latent_dim, self.d_model)
        self.input_projection = nn.Linear(3, self.d_model)
        self.output_projection = nn.Linear(self.d_model, 3)
        self.emb1 = nn.Embedding(9, self.d_model)
        self.emb2 = nn.Embedding(9, self.d_model)
        self.aa_emb = nn.Embedding(21, self.d_model)
        self.ss_emb = nn.Embedding(4, self.d_model)

        # Initialize InputAttention module
        self.input_attention = InputAttention(
            input_dim=self.d_model,
            mlp_hidden_dim=64,
            orientation_dim=self.d_model,
            embedding_dim=self.d_model,
            attention_dim=self.d_model,
            output_dim=self.d_model
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_weights)
        self.angles = []

    def encode(self, x, labels, displacement):
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        x, first_three = self.c2d(x)
        x = x.unflatten(1,(14,3))
        x = self.input_projection(x)
        # input_mu, input_sigma = self.input_mu(displacement).unsqueeze(1), self.input_sigma(displacement).unsqueeze(1)
        input_mu, input_sigma = torch.ones(x.shape[0], 1, self.d_model).to(x.device), torch.ones(x.shape[0], 1, self.d_model).to(x.device)
        x = torch.cat([input_mu, input_sigma, x], dim=1)
        x = self.positional_encoding(x)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        mean = self.output_mu(x[:,0,:])
        log_variance = self.output_sigma(x[:,1,:])
        return mean, log_variance, first_three

    def sample(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std, device=std.device)
        return mean + std * epsilon

    def decode(self, x, labels, displacement, first_three, return_angles=False, train_data=None):
        aa, ss = labels[:,1:,0].long(), labels[:,1:,1].long()
        displacement_1 = torch.stack([torch.arccos(displacement[:,1]/(1e-6+torch.sqrt(1e-6+displacement[:,0]**2 + displacement[:,1]**2) + displacement[:,2]**2)),
        torch.sign(displacement[:,1]) * torch.arccos(displacement[:,0]/(1e-6+torch.sqrt(1e-6+displacement[:,0]**2 + displacement[:,1]**2)))], dim=-1)
        boundaries = torch.tensor([-2*math.pi, -3*math.pi/2, -math.pi, -math.pi/2, 0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi], device=displacement.device)
        displacement_1 = torch.bucketize(displacement_1, boundaries)
        x = self.decoder_projection(x)
        x = x.unsqueeze(1).expand(-1,14,-1)

        # Apply InputAttention module
        tgt, _ = self.input_attention(x, self.emb1(displacement_1[:,0]) + self.emb2(displacement_1[:,1]))
        tgt = self.positional_encoding(tgt) + self.aa_emb(aa) + self.ss_emb(ss)

        x = self.decoder(inputs_embeds=tgt, encoder_hidden_states= self.emb1(displacement_1[:,0]).unsqueeze(1) + self.emb2(displacement_1[:,1]).unsqueeze(1)).last_hidden_state
        x = self.output_projection(x)
        x = x.flatten(1,2)
        x = torch.tanh(x) * torch.full((x.shape[0],42), math.pi, device=self.device)
        x = self.d2c((x, first_three),return_angles=return_angles, train_data=train_data)
        return x

    def displacement_loss(self, recon_x_disp, displacement):
        alpha = 1.0  # You can adjust this value as needed
        beta = 1.0   # You can adjust this value as needed
        cos_sim = F.cosine_similarity(recon_x_disp, displacement)
        loss = torch.log(1 + torch.exp(-alpha * (cos_sim - beta)))
        return torch.mean(loss)

    def loss(self, recon_x, recon_x_disp, x, mean, log_variance, displacement, first_three, weights):
        beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (math.cos(math.pi * self.current_epoch/10))
        recon_loss_1 = torch.mean(F.mse_loss(recon_x, x.flatten(1,2)[:,3:,:], reduction='none'))
        recon_loss_2 = torch.mean(F.mse_loss(recon_x_disp, x.flatten(1,2)[:,3:,:], reduction='none'))
        kl_loss = torch.mean(-0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp()))
        displacement_loss = torch.mean(F.mse_loss(recon_x_disp[:,-1,:] - first_three[:,-1,:], displacement, reduction='none'))
        # recon_weight = self.current_epoch % 2 == 0
        recon_weight = 0
        loss = recon_weight * 1 * recon_loss_1 + (1 - recon_weight) * .1 * recon_loss_2 + beta*kl_loss + 1e-2*displacement_loss
        return loss, recon_loss_1, recon_loss_2, kl_loss, displacement_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        inputs_cart, labels = batch
        weights = labels[:,-1,2].unsqueeze(1).unsqueeze(2)
        labels = labels[:,:,(0,1,3,4,5)]

        displacement = inputs_cart[:,-1,-1,:] - inputs_cart[:,0,-1,:]

        mean, log_variance, first_three = self.encode(inputs_cart, labels, displacement)

        z = self.sample(mean, log_variance)
        recon_inputs = self.decode(z, labels, displacement, first_three, train_data=inputs_cart[:,1:,:,:].flatten(1,2))
        recon_inputs_disp = self.decode(z, labels, displacement, first_three)

        loss, recon_loss_1, recon_loss_2, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs_disp, inputs_cart, mean, log_variance, displacement, first_three, weights)

        z_1 = torch.randn_like(z)
        x = self.decode(z_1, labels, displacement, first_three)
        d_loss = F.mse_loss(x[:,-1,:] - first_three[:,-1,:], displacement)
        loss += 0.01 * d_loss
        self.log("d_loss", d_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_recon_loss_1", recon_loss_1, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_recon_loss_2", recon_loss_2, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_displacement", displacement_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_displacement_1", torch.mean(F.mse_loss(recon_inputs_disp[:,-1,:] - first_three[:,-1,:], displacement, reduction='none')), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs_cart, labels = batch
        weights = labels[:,-1,2].unsqueeze(1).unsqueeze(2)
        labels = labels[:,:,(0,1,3,4,5)]

        displacement = inputs_cart[:,-1,-1,:] - inputs_cart[:,0,-1,:]

        mean, log_variance, first_three = self.encode(inputs_cart, labels, displacement)
        z = self.sample(mean, log_variance)
        recon_inputs = self.decode(z, labels, displacement, first_three)

        loss, recon_loss, _, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs, inputs_cart, mean, log_variance, displacement, first_three, weights)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement", displacement_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement_1", torch.mean(F.mse_loss(recon_inputs[:,-1,:] - first_three[:,-1,:], displacement, reduction='none')), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        recon_inputs, angles = self.generate(inputs_cart.shape[0], first_three, labels, displacement, return_angles=True)
        loss, recon_loss, _, kl_loss, displacement_loss = self.loss(recon_inputs, recon_inputs, inputs_cart, mean, log_variance, displacement, first_three, weights)
        below_1 = torch.mean((displacement_loss < 1.0).float())
        self.log("val_recon_loss_generation", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_displacement_generation", displacement_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_below_1_generation", below_1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("generation_displacement_1", torch.mean(F.mse_loss(recon_inputs[:,-1,:] - first_three[:,-1,:], displacement, reduction='none')), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.angles.append(angles)

    def generate(self, n, first_three, labels, displacement, return_angles=False):
        z = torch.randn((n, self.latent_dim), device=displacement.device)
        return self.decode(z, labels, displacement, first_three, return_angles=return_angles)

    def on_validation_end(self):
        if len(self.angles) == 0:
            return
        angles = torch.cat(self.angles)
        print(angles)

        fig, ax = plt.subplots()
        ax.hist2d(angles[:,2].cpu().detach().numpy(), angles[:,0].cpu().detach().numpy(), bins=[300,300], range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap="Blues", norm=colors.LogNorm())
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
            latent_dim=self.latent_dim,
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
