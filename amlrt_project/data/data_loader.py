import logging
import typing
from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

from amlrt_project.data.data_preprocess import FashionMnistParser
from amlrt_project.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)
# __TODO__ change the dataloader to suit your needs...


class FashionMnistDS(Dataset):  # pragma: no cover
    """Dataset class for iterating over the data."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        """Initialize Dataset.

        Args:
            images (np.array): Image data [batch, height, width].
            labels (np.array): Target data [batch,].
            transform (Callable[[torch.tensor], torch.tensor], optional): Valid tensor transformations.  # noqa
            Defaults to None.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the number of data items in Dataset."""
        return len(self.images)

    def __getitem__(
        self,
        idx: int,
    ):
        """__getitem__.

        Args:
            idx (int): Get index item from the dataset.
        """
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class FashionMnistDM(pl.LightningDataModule):  # pragma: no cover
    """Data module class that prepares dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        check_and_log_hp(["batch_size", "num_workers"], hyper_params)
        self.data_dir = data_dir
        self.batch_size = hyper_params["batch_size"]
        self.num_workers = hyper_params["num_workers"]

    def setup(self, stage=None):
        """Parses and splits all samples across the train/valid/test parsers."""
        # here, we will actually assign train/val datasets for use in dataloaders
        raw_data = FashionMnistParser(data_dir=self.data_dir)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = FashionMnistDS(
                raw_data.train_images, raw_data.train_labels, transform
            )
            self.val_dataset = FashionMnistDS(
                raw_data.val_images, raw_data.val_labels, transform
            )
        if stage == "test" or stage is None:
            self.test_dataset = FashionMnistDS(
                raw_data.test_images, raw_data.test_labels, transform
            )

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Creates the validation dataloader using the validation data parser."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class FashionMnistTVDM(pl.LightningDataModule):
    """Lightning DataModule backed by torchvision.datasets.FashionMNIST.

    Constructor kept compatible with the original cookiecutter version:
        __init__(data_dir, hyper_params)
    Required hyper_params: batch_size, num_workers
    Optional hyper_params:
        - val_split (int, default=5000)
        - normalize (tuple(mean, std), default=(0.2860, 0.3530))
        - pin_memory (bool, default=True)
        - persistent_workers (bool, default=True)
        - shuffle (bool, default=True)  # train loader
        - drop_last (bool, default=False)  # train loader
        - seed (int, default=42)  # for deterministic split
    """

    def __init__(
            self, 
            data_dir: typing.AnyStr, 
            hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        check_and_log_hp(["batch_size", "num_workers"], hyper_params)

        self.data_dir: str = str(data_dir)
        self.batch_size: int = int(hyper_params["batch_size"])
        self.num_workers: int = int(hyper_params["num_workers"])

        # Optional knobs with sensible defaults (all pulled from hyper_params if present)
        self.val_split: int = int(hyper_params.get("val_split", 5000))
        mean, std = hyper_params.get("normalize", (0.2860, 0.3530))
        self.normalize: Tuple[float, float] = (float(mean), float(std))

        self.pin_memory: bool = bool(hyper_params.get("pin_memory", True))
        self.persistent_workers: bool = bool(hyper_params.get("persistent_workers", True))
        self.shuffle: bool = bool(hyper_params.get("shuffle", True))
        self.drop_last: bool = bool(hyper_params.get("drop_last", False))
        self.seed: int = int(hyper_params.get("seed", 42))

        # Will be populated in setup()
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.normalize[0],), (self.normalize[1],))
        ])

    # ---- Lightning hooks ----
    def prepare_data(self) -> None:
        """Download once on a single process."""
        datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets/splits on every process."""
        if stage in (None, "fit"):
            full_train = datasets.FashionMNIST(
                root=self.data_dir, train=True, transform=self.transform, download=False
            )
            if self.val_split > 0:
                train_len = len(full_train) - self.val_split
                # deterministic split
                self.train_set, self.val_set = random_split(
                    full_train,
                    [train_len, self.val_split],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                self.train_set = full_train
                self.val_set = None

        if stage in (None, "test", "validate"):
            self.test_set = datasets.FashionMNIST(
                root=self.data_dir, train=False, transform=self.transform, download=False
            )

    # ---- Dataloaders ----
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    

def load_dataset(data_dir: str, hyper_params: dict) -> pl.LightningDataModule:
    """
    Select the between torch vision pre-defined data class and the original one.
    """
    dataclass = hyper_params.get("dataclass", "").lower()

    match dataclass:
        case "fashionmnisttvdm":
            datamodule = FashionMnistTVDM(data_dir, hyper_params)
            logger.info('Using FashionMnistTVDM')
        case _: 
            datamodule = FashionMnistDM(data_dir, hyper_params)
            logger.info('Using FashionMnistDM')

    return datamodule