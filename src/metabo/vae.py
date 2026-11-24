"""VAE as a scikit-learn transformer (encoder 256-64-k, decoder k-64-256).

Early stopping on validation reconstruction loss, so the label is never used. The best
weights are restored. transform() returns the latent mean. Uses CUDA when available.
torch is imported only when the VAE is fitted.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from metabo import config


class VAEEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, latent_dim=2, hidden=(256, 64), lr=9e-5,
                 max_epochs=None, patience=None, val_fraction=None,
                 batch_size=128, kappa=1e-4, random_state=42):
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.lr = lr
        self.max_epochs = config.VAE_MAX_EPOCHS if max_epochs is None else max_epochs
        self.patience = config.VAE_PATIENCE if patience is None else patience
        self.val_fraction = config.VAE_VAL_FRACTION if val_fraction is None else val_fraction
        self.batch_size = batch_size
        self.kappa = kappa
        self.random_state = random_state

    def _build(self, n_feat):
        import torch
        import torch.nn as nn
        h1, h2 = self.hidden
        ld = self.latent_dim

        class _VAE(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(nn.Linear(n_feat, h1), nn.ReLU(),
                                      nn.Linear(h1, h2), nn.ReLU())
                s.mu = nn.Linear(h2, ld)
                s.logvar = nn.Linear(h2, ld)
                s.dec = nn.Sequential(nn.Linear(ld, h2), nn.ReLU(),
                                      nn.Linear(h2, h1), nn.ReLU(),
                                      nn.Linear(h1, n_feat))

            def encode(s, x):
                h = s.enc(x)
                return s.mu(h), s.logvar(h)

            def reparam(s, mu, lv):
                std = torch.exp(0.5 * lv)
                return mu + torch.randn_like(std) * std

            def forward(s, x):
                mu, lv = s.encode(x)
                z = s.reparam(mu, lv)
                return s.dec(z), mu, lv

        return _VAE()

    def fit(self, X, y=None):
        import copy
        import torch
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.random_state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = np.asarray(X, dtype="float32")

        x_tr, x_val = train_test_split(X, test_size=self.val_fraction,
                                       random_state=self.random_state)
        x_tr = torch.tensor(x_tr).to(device)
        x_val = torch.tensor(x_val).to(device)

        self.model_ = self._build(X.shape[1]).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()

        best_val = np.inf
        best_state = None
        bad = 0
        n = x_tr.shape[0]
        for _ in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                xb = x_tr[perm[i:i + self.batch_size]]
                xhat, mu, lv = self.model_(xb)
                kl = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))
                loss = mse(xhat, xb) + self.kappa * kl
                opt.zero_grad()
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                xhat_val, _, _ = self.model_(x_val)
                val_loss = mse(xhat_val, x_val).item()
            if val_loss < best_val - 1e-5:
                best_val, bad = val_loss, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = float(best_val)
        self.device_ = device
        return self

    def transform(self, X):
        import torch
        self.model_.eval()
        X = np.asarray(X, dtype="float32")
        with torch.no_grad():
            mu, _ = self.model_.encode(torch.tensor(X).to(self.device_))
        return mu.cpu().numpy()
