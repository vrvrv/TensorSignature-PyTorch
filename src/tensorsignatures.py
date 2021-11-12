import torch
import pickle
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from scipy.stats import truncnorm


def truncated_normal(size, threshold=2):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return torch.from_numpy(values)


def deviance(pred, true):
    d_ij = 2 * (pred * torch.log(pred / (true + 1e-8) + 1e-8) - pred + true)
    return d_ij.mean()


class TensorSignature(pl.LightningModule):
    def __init__(self,
                 rank,
                 snv_shape,
                 size=50,
                 objective='nbconst',
                 starter_learning_rate=0.1
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparams
        self.samples = self.hparams.snv_shape[-1]

        # dimensions
        self.p = self.hparams.snv_shape[-2]

        # initialize C1 and C2
        self.register_buffer(
            'tau', torch.Tensor([self.hparams.size])
        )
        # Clustering dims
        self.c = len(self.hparams.snv_shape) - 4
        self.card = list(self.hparams.snv_shape)[2: -2]
        self.card_prod = np.prod(self.card)
        self.idex = self.indices_to_assignment(
            np.arange(self.card_prod), self.card
        )

        # Define parameters
        self.S0 = nn.Parameter(
            truncated_normal([2, 2, self.p - 1, self.hparams.rank])
        )
        self.E0 = nn.Parameter(
            truncated_normal([self.hparams.rank, self.samples])
        )
        self.a0 = nn.Parameter(
            truncated_normal([2, self.hparams.rank])
        )
        self.b0 = nn.Parameter(
            truncated_normal([2, self.hparams.rank])
        )

        for i in range(2, 2 + self.c):
            k = i - 2
            setattr(
                self, 'k{}'.format(k),
                nn.Parameter(
                    truncated_normal([self.card[k] - 1, self.hparams.rank])
                ),
            )

        self.m0 = nn.Parameter(
            truncated_normal([1, self.hparams.rank])
        )

    @property
    def S1(self):
        S0s = torch.softmax(
            torch.cat([self.S0, torch.zeros([2, 2, 1, self.hparams.rank], device=self.device)], dim=2), dim=2
        )

        _S1 = torch.stack([
            S0s[0, 0, :, :], S0s[1, 0, :, :], 0.5 * S0s[:, 0, :, :].sum(0),
            S0s[1, 1, :, :], S0s[0, 1, :, :], 0.5 * S0s[:, 1, :, :].sum(0),
                                              0.5 * (S0s[0, 0, :, :] + S0s[1, 1, :, :]),
                                              0.5 * (S0s[1, 0, :, :] + S0s[0, 1, :, :]),
                                              0.25 * S0s.sum(dim=(0, 1))
        ]).reshape(3, 3, 1, self.p, self.hparams.rank)

        return _S1

    def E(self, idx):
        return torch.exp(self.E0[..., idx])

    @property
    def A(self):
        a1 = torch.exp(
            torch.cat([self.a0, self.a0, torch.zeros((2, self.hparams.rank), device=self.device)], dim=0)
        ).reshape(3, 2, self.hparams.rank)

        _A = (a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :]).reshape(
            3, 3, 1, 1, self.hparams.rank
        )

        return _A

    @property
    def B(self):
        _B = torch.exp(
            torch.stack([
                self.b0[0, :] + self.b0[1, :], self.b0[0, :] - self.b0[1, :], self.b0[0, :],
                self.b0[1, :] - self.b0[0, :], -self.b0[1, :] - self.b0[0, :], -self.b0[0, :],
                self.b0[1, :], -self.b0[1, :], torch.zeros(self.b0[0, :].shape, device=self.device)
            ]).reshape(3, 3, 1, 1, self.hparams.rank)
        )

        return _B

    @property
    def K(self):

        _cbiases = {}

        for i in range(2, 2 + self.c):
            k = i - 2
            v = getattr(self, 'k{}'.format(k))

            _cbiases[k] = torch.cat(
                [torch.zeros([1, self.hparams.rank], device=self.device), v], dim=0
            )

        final_tensor = []
        for r in range(self.idex.shape[0]):
            current_term = []
            for c in range(self.idex.shape[1]):
                current_term.append(
                    _cbiases[c][self.idex[r, c].astype(int), :]
                )
            final_tensor.append(
                torch.stack(current_term).sum(dim=0)
            )

        _K = torch.exp(
            torch.stack(final_tensor).reshape(1, 1, -1, 1, self.hparams.rank)
        )

        return _K

    @property
    def M(self):
        return torch.sigmoid(self.m0).reshape(1, 1, 1, 1, self.hparams.rank)

    @property
    def S(self):
        return self.S1 * self.A * self.B * self.K * self.M

    def Chat(self, idx):
        return torch.matmul(
            self.S.reshape(-1, self.hparams.rank), self.E(idx)
        ).reshape(3, 3, -1, self.p, len(idx))

    def Lij(self, C, idx):

        Chat = self.Chat(idx)

        if self.hparams.objective == 'nbconst':
            _Lij = self.tau \
                   * torch.log(self.tau) \
                   - torch.lgamma(self.tau) \
                   + torch.lgamma(C + self.tau) \
                   + C * torch.log(Chat) \
                   - torch.log(Chat + self.tau) \
                   * (self.tau + C) \
                   - torch.lgamma(C + 1)

        elif self.hparams.objective == 'poisson':
            _Lij = C \
                   * torch.log(Chat) \
                   - Chat \
                   - torch.lgamma(C + 1)

        return _Lij

    def forward(self, C, idx):
        return - self.Lij(C, idx).sum() / len(idx)

    def training_step(self, batch, batch_idx):
        C, idx = batch
        loss = self(C, idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     C, idx = batch
    #     loss = self(C, idx)
    #     self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        C, idx = batch
        Chat = self.Chat(idx)

        dev = deviance(Chat, C)
        self.log("deviance", dev, logger=True)

    def indices_to_assignment(self, I, card):
        # Helper function to collapse additional genomic dimension
        card = np.array(card, copy=False)
        C = card.flatten()
        A = np.mod(
            np.floor(
                np.tile(I.flatten().T, (len(card), 1)).T /
                np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))),
                        (len(I), 1))),
            np.tile(C[::-1], (len(I), 1)))

        return A[:, ::-1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.starter_learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95 ** (1 / 1000))}
        }
