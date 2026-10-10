import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(
        self, 
        in_dim, enc_hidden_dim, dec_hidden_dim,
        latent_dim, conditional=False, condition_dim=None
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.conditional = conditional

        if self.conditional and condition_dim is not None:
            input_dim = in_dim + condition_dim
            dec_dim = latent_dim + condition_dim
        else:
            input_dim = in_dim
            dec_dim = latent_dim
        self.enc_MLP = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            nn.ELU())
        self.linear_means = nn.Linear(enc_hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(enc_hidden_dim, latent_dim)
        self.dec_MLP = nn.Sequential(
            nn.Linear(dec_dim, dec_hidden_dim),
            nn.ELU(),
            nn.Linear(dec_hidden_dim, in_dim))

    def forward(self, x, c=None, return_pred=False):
        if self.conditional and c is not None:
            inp = torch.cat((x, c), dim=-1)
        else:
            inp = x
        h = self.enc_MLP(inp)
        mean = self.linear_means(h)
        # print("Mean:", mean.shape)

        log_var = self.linear_log_var(h)
        # print("Log var:", log_var.shape)
        z = self.reparameterize(mean, log_var)
        # print("z shape:", z.shape)
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        recon_loss, KLD = self.loss_fn(recon_x, x, mean, log_var)
        # print("KLD:", KLD.shape)
        if not return_pred:
            return recon_loss, KLD
        else:
            return recon_x, recon_loss, KLD

    def loss_fn(self, recon_x, x, mean, log_var):
        recon_loss = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None, x=None, return_kl=False):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)

        if return_kl:
            assert x is not None
            if self.conditional and c is not None:
                inp = torch.cat((x, c), dim=-1)
            else:
                inp = x
            h = self.enc_MLP(inp)
            mean = self.linear_means(h)

            log_var = self.linear_log_var(h)
            # print("Log var:", log_var.shape)
            z = self.reparameterize(mean, log_var)
            # print("z shape:", z.shape)
            if self.conditional and c is not None:
                z = torch.cat((z, c), dim=-1)
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
            return recon_x, KLD
        return recon_x