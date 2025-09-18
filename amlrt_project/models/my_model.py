import logging
import typing

import pytorch_lightning as pl
from torch import FloatTensor, LongTensor, nn

from amlrt_project.models.factory import (OptimFactory,
                                          OptimizerConfigurationFactory,
                                          SchedulerFactory)
from amlrt_project.models.optim import load_loss
from amlrt_project.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optim_fact: OptimFactory,
        sched_fact: SchedulerFactory,
    ):
        """Initialize the LightningModule, with the actual model and loss."""
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.opt_fact = OptimizerConfigurationFactory(optim_fact, sched_fact)

    def configure_optimizers(self):
        """Returns the combination of optimizer(s) and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.

        This function will be called automatically by the pytorch lightning trainer implementation.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html for more info
        on the expected returned elements.
        """
        return self.opt_fact(self.model.parameters())

    def forward(self, input_data: FloatTensor) -> FloatTensor:
        """Invoke the model."""
        return self.model(input_data)

    def _generic_step(
            self,
            batch: typing.Tuple[FloatTensor, LongTensor],
            batch_idx: int,
    ) -> FloatTensor:
        """Runs the prediction + evaluation step for training/validation/testing."""
        input_data, targets = batch
        preds = self(input_data)  # calls the forward pass of the model
        loss = self.loss_fn(preds, targets)
        return loss

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return loss  # this function is required, as the loss returned here is used for backprop

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("test_loss", loss)


class SimpleMLP(BaseModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """
    def __init__(
        self,
        optim_fact: OptimFactory,
        sched_fact: SchedulerFactory,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ):
        """__init__.

        Args:
            optim_fact (OptimFactory): factory for the optimizer.
            sched_fact (SchedulerFactory): factory for the scheduler.
            hyper_params (dict): hyper parameters from the config file.
        """
        # TODO: Place this in a factory.
        check_and_log_hp(['hidden_dim', 'num_classes'], hyper_params)
        num_classes: int = hyper_params['num_classes']
        hidden_dim: int = hyper_params['hidden_dim']
        flatten = nn.Flatten()
        mlp_layers = nn.Sequential(
            nn.Linear(
                784, hidden_dim,
            ),  # The input size for the linear layer is determined by the previous operations
            nn.ReLU(),
            nn.Linear(
                hidden_dim, num_classes
            ),  # Here we get exactly num_classes logits at the output
        )
        model = nn.Sequential(flatten, mlp_layers)

        super().__init__(
            model, load_loss(hyper_params),
            optim_fact, sched_fact)
        self.save_hyperparameters()  # they will become available via model.hparams


class SimpleCNN(BaseModel):  # pragma: no cover
    """Simple CNN model class.

    Mirrors SimpleMLP style: builds a torch.nn.Module, then calls BaseModel
    with loss/optim/sched factories, and stores hyperparams via save_hyperparameters().
    """

    def __init__(
        self,
        optim_fact: OptimFactory,
        sched_fact: SchedulerFactory,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        # Required hyper-params for this CNN
        # (You can add more, e.g., dropout/pooling, with defaults if desired.)
        check_and_log_hp(
            ['in_channels', 'hidden_channels', 'kernel_size', 'num_classes', 'input_hw'],
            hyper_params,
        )

        in_channels: int = int(hyper_params['in_channels'])
        hidden_channels: int = int(hyper_params['hidden_channels'])
        kernel_size: int = int(hyper_params['kernel_size'])
        num_classes: int = int(hyper_params['num_classes'])
        # e.g., [28, 28] for MNIST, [32, 32] for CIFAR-10
        H, W = map(int, hyper_params['input_hw'])

        # same-padding for stride=1 convs
        pad = kernel_size // 2

        # Feature extractor: 2 conv blocks + 2×2 pool each
        features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # After two 2×2 pools: H_out = H//4, W_out = W//4
        h_out, w_out = H // 4, W // 4
        fc_in = (hidden_channels * 2) * h_out * w_out

        classifier = nn.Sequential(
            nn.Linear(fc_in, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, num_classes),
        )

        model = nn.Sequential(features, classifier)

        super().__init__(model, load_loss(hyper_params), optim_fact, sched_fact)
        self.save_hyperparameters()  # makes them available as model.hparams
