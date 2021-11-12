import torch
import pickle

from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


def collate_fn(batch):
    snv = []
    idx = []
    for snv_i, i in batch:
        snv.append(
            torch.from_numpy(snv_i)
        )
        idx.append(i)

    snv = torch.stack(snv, dim=-1)
    idx = torch.tensor(idx, dtype=torch.long)

    return snv, idx


class SNVDataset(Dataset):

    def __init__(self, data_dir: str):
        super().__init__()
        with open(data_dir, "rb") as f:
            self.snv = pickle.load(f)

    def __len__(self):
        return self.snv.shape[-1]

    def __getitem__(self, idx):
        return self.snv[..., idx].reshape(3, 3, -1, 96), idx


4


class emptyDataModule(LightningDataModule):
    def __init__(self,
                 name: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int
                 ):
        super(emptyDataModule, self).__init__()
        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        SNV = SNVDataset(data_dir=self.hparams.data_dir)

        # self.train, self.val = random_split(
        #     SNV, [int(len(SNV) * 0.9), len(SNV) - int(len(SNV) * 0.9)]
        # )

        self.train = self.test = SNV

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            shuffle=True
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         dataset=self.val,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         collate_fn=collate_fn,
    #         shuffle=False
    #     )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=len(self.test),
            num_workers=1,
            collate_fn=collate_fn,
            shuffle=False
        )
