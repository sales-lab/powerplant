# PowerPlant: Deep learning framework for the classification of herbarium samples
# Copyright (C) 2025 Gabriele Sales
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

from powerplant.dataload import (
    Counters,
    DataLoaders,
    compute_counters,
    create_dataloaders,
)
from powerplant.model import RegressionModel
from powerplant.resize import NoImagesFound, resize_images
from powerplant.segment import MissingMask, generate_masks, mask_images
from powerplant.train import train_model
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import LinearLR

import torch.nn as nn
import logging


def segment_cmd():
    setup_logging()

    input_dir = "images/original"
    resized_dir = "images/resized"
    mask_dir = "images/masks"
    masked_dir = "images/masked"

    try:
        resize_images(input_dir, resized_dir, 2048)
    except NoImagesFound:
        exit(f"No images found. Did you save them to {input_dir!r}?")

    generate_masks(resized_dir, mask_dir)

    try:
        mask_images(input_dir, mask_dir, masked_dir, 1024, 4000, 0.02)
    except MissingMask as e:
        exit(e.args[0])


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_cmd():
    counters = compute_counters("metadata/samples.csv")
    model = setup_model(counters)
    loaders = setup_dataloaders(model)
    _train(model, loaders)

    dest = "checkpoints/prediction/trained.pth"
    model.save_state(dest)
    print(f"✅ Saved trained model in: {dest}")


def setup_model(counters: Counters) -> nn.Module:
    backend = "timm/res2net50d.in1k"
    embed_dim = 512
    layer_num = 2
    head_num = 4
    ff_dim = 512
    dropout = 0.432
    regr_dim = 768

    return RegressionModel(
        counters.orders,
        counters.clades,
        counters.countries,
        embed_dim=embed_dim,
        layer_num=layer_num,
        use_coords=True,
        head_num=head_num,
        backend_name=backend,
        feedforward_dim=ff_dim,
        transformer_dropout=dropout,
        regr_dim=regr_dim,
    )


def setup_dataloaders(model: nn.Module) -> DataLoaders:
    datadir = "dataset"
    crop_size = 896
    cut_size = 320
    batch_size = 4

    return create_dataloaders(
        datadir,
        model,
        crop_size=crop_size,
        cut_size=cut_size,
        batch_size=batch_size,
        eval_set="test",
        num_workers=0,
    )


def _train(model: nn.Module, loaders: DataLoaders) -> None:
    grad_accum_steps = 2
    epochs = 16
    lr = 5.41e-05
    lr_end = 0.034
    optimizer = create_optimizer_v2(model.parameters(), opt="adamw", lr=lr)
    scheduler = LinearLR(
        optimizer, start_factor=1, end_factor=lr_end, total_iters=epochs
    )
    criterion = nn.MSELoss(reduction="sum")

    train_model(
        model,
        optimizer,
        criterion,
        loaders.train,
        loaders.eval,
        epochs=epochs,
        scheduler=scheduler,
        gradient_accum_steps=grad_accum_steps,
        load_best=True,
    )
