# PowerPlant: Deep learning framework for the classification of herbarium samples
# Copyright (C) 2025  Gabriele Sales
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from functools import partial
from inspect import signature
from os.path import join
from tempfile import TemporaryDirectory
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from typing import Callable, Iterable, Optional

import numpy as np
import torch


@dataclass
class ModelPerf:
    epoch: int
    loss: float
    pearson: float
    eval_truth: np.array
    eval_preds: np.array


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    device: str = "cpu",
    scheduler: Optional[LRScheduler] = None,
    gradient_accum_steps: int = 1,
    load_best: bool = True,
    show_progress: bool = True,
) -> ModelPerf:
    """
    Train a model using the provided optimizer, performance criterion and
    data loaders.

    Args:
        model (Module): The model to train.
        optimizer (Optimizer): An optimization algorithm.
        criterion (Module): A module computing the training loss.
        train_loader (DataLoader): Loader for training samples.
        eval_loader (DataLoader): Loader for evaluation samples.
        epochs (int): Number of training epochs.
        device (str, optional): Compute device to use for training. Defaults to
            "cpu".
        scheduler (LRScheduler, optional): Learning rate scheduler. Default to
            None.
        gradient_accum_steps (int, optional): The number of steps over which to
            accumulate the gradient. Defaults to 1.
        load_best (bool, optional): Load the best weights at the end of
            training. Defaults to True.
        show_progress (bool, optional): Show bars to track training progress.
            Default to True.

    Returns:
        ModelPerf: Validation metrics of the model at the best epoch.
    """
    if criterion.reduction != "sum":
        raise ValueError("the loss reduction should be set to 'sum'")

    def mk_epochs():
        if show_progress:
            return trange(epochs, desc="Training epoch")
        else:
            return range(epochs)

    def mk_train():
        if show_progress:
            return tqdm(train_loader, desc="Training batch", leave=False)
        else:
            return train_loader

    def mk_eval():
        if show_progress:
            return tqdm(eval_loader, desc="Evaluation batch", leave=False)
        else:
            return eval_loader

    loop = partial(
        _loop,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        gradient_accum_steps=gradient_accum_steps,
        mk_epochs=mk_epochs,
        mk_train=mk_train,
        mk_eval=mk_eval,
        scheduler=scheduler,
        show_perf=show_progress,
    )
    if load_best:
        with TemporaryDirectory(dir=".") as checkpoint_dir:
            best = loop(checkpoint_dir=checkpoint_dir)
    else:
        best = loop(checkpoint_dir=None)

    if show_progress:
        print(
            f"Best epoch: epoch={best.epoch}  loss={best.loss:.4f}  pearson={best.pearson:.4f}"
        )

    return best


_MakeIter = Callable[[], Iterable]


def _loop(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: str,
    gradient_accum_steps: int,
    mk_epochs: _MakeIter,
    mk_train: _MakeIter,
    mk_eval: _MakeIter,
    show_perf: bool,
    scheduler: Optional[LRScheduler],
    checkpoint_dir: Optional[str],
) -> ModelPerf:
    best = ModelPerf(0, np.inf, 0, np.array([]), np.array([]))
    checkpoint = join(checkpoint_dir, "weights.pth") if checkpoint_dir else None

    for epoch in mk_epochs():
        model.train()

        train_loss = 0
        train_steps = 0
        accum_num = 0

        for batch in mk_train():
            if accum_num == 0:
                optimizer.zero_grad()

            feats = {f: vs.to(device) for f, vs in batch.items() if f != "barcode"}
            outputs = model(**feats)

            loss = criterion(outputs, feats["concentr"])
            loss.backward()

            train_loss += loss.item()
            train_steps += len(feats["concentr"])
            accum_num += 1

            if accum_num >= gradient_accum_steps:
                optimizer.step()
                accum_num = 0

        if accum_num > 0:
            optimizer.step()

        model.eval()
        eval_loss = 0
        eval_steps = 0
        y_true = []
        y_pred = []
        for batch in mk_eval():
            feats = {f: vs.to(device) for f, vs in batch.items() if f != "barcode"}

            with torch.no_grad():
                outputs = model(**feats)

            loss = criterion(outputs, feats["concentr"])
            eval_loss += loss.item()
            eval_steps += len(feats["concentr"])
            y_true.extend(feats["concentr"].cpu().numpy().flatten())
            preds = outputs["predictions"] if isinstance(outputs, dict) else outputs
            y_pred.extend(preds.cpu().numpy().flatten())

        train_loss_100 = train_loss / train_steps * 100
        eval_loss_100 = eval_loss / eval_steps * 100

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]

        if show_perf:
            print(
                f"{epoch=:<2}  {train_loss_100=:.4f}  {eval_loss_100=:.4f}  {pearson=:.4f}"
            )

        if eval_loss_100 < best.loss:
            best = ModelPerf(epoch, eval_loss_100, pearson, y_true, y_pred)

            if checkpoint:
                torch.save(model.state_dict(), checkpoint)

        if scheduler is not None:
            if len(signature(scheduler.step).parameters) == 2:
                scheduler.step(loss)
            else:
                scheduler.step()

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))

    return best
