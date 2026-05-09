import pyrootutils
import os

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
import pytorch_lightning as pl
import torch

from src.dataio import akwd_dataset
# Optional
from typing import Optional

# LightningDataModuleはDataLoaderとなるクラス


class AWKDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, predict_dir: Optional[str] = None):
        super().__init__()  # 親クラスのinit
        self.dataset = akwd_dataset.AKWDDataset(root=data_dir)
        if predict_dir is not None:
            self.predict_dataset = akwd_dataset.AKWDDataset(root=predict_dir)

        # If the dataset size differs from the original expected count, split automatically.
        dataset_len = len(self.dataset)
        if dataset_len < 3:
            raise ValueError(f"Dataset too small to split: {dataset_len}")

        num_train = int(dataset_len * 0.8)
        remaining = dataset_len - num_train
        num_val = remaining // 2
        num_test = dataset_len - num_train - num_val

        print(
            f"Dataset size: {dataset_len}, train={num_train}, val={num_val}, test={num_test}"
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.dataset, [num_train, num_val, num_test]
        )

        self.batch_size = batch_size
        self.data_dir = data_dir

    def train_dataloader(self):  # Train用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def val_dataloader(self):  # val用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self):  # Test用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


if __name__ == "__main__":
    data_module = AWKDDataModule(batch_size=32, data_dir=root / "data/AKWF_44k1_600s")
    print(data_module.train_dataloader())
    print(data_module.val_dataloader())
    print(data_module.test_dataloader())
    print(data_module.predict_dataloader())
