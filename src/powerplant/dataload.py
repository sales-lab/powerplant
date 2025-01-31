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
from datasets import (
    Dataset,
    load_dataset,
    are_progress_bars_disabled,
    enable_progress_bars,
    disable_progress_bars,
)
from powerplant.model import BaseModule
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import Dict, Optional

import torch


@dataclass
class DataLoaders:
    train: DataLoader
    eval: DataLoader


class PadImage:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        size = self.min_size

        channels, height, width = img.size()
        if width >= size and height >= size:
            return img

        h2 = max(size, height)
        w2 = max(size, width)

        padded = (
            torch.ones((channels, h2, w2), dtype=img.dtype, device=img.device) * 255
        )
        x = (w2 - width) // 2
        y = (h2 - height) // 2
        padded[:, y : y + height, x : x + width] = img

        return padded


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        side = self.crop_size

        height, width = img.size()[1:]
        x = torch.randint(
            0, width - side + 1, (1,), dtype=torch.int32, device=img.device
        ).item()
        y = torch.randint(
            0, height - side + 1, (1,), dtype=torch.int32, device=img.device
        ).item()
        return img[:, y : y + side, x : x + side]


class FocusCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        c = self.crop_size
        w = img.size(2)

        n = w // c
        if w % c > 0:
            n += 1

        mp = img[0]
        mp = mp.unfold(0, c, c)
        mp = mp.unfold(1, c, c)
        mp = torch.sqrt(torch.sum(mp < 255, dim=(2, 3), dtype=torch.float32))
        mp = mp.view(-1)

        if not torch.any(mp > 0):
            sel = 0
        else:
            sel = torch.max(mp, 0).indices.item()

        x = sel // n * c
        y = sel % n * c
        return img[:, y : y + c, x : x + c]


class RandomCut:
    def __init__(self, size):
        self.cut_size = size

    def __call__(self, img):
        height, width = img.size()[1:]
        size = min(min(height, width) // 2, self.cut_size)

        x = torch.randint(0, width - size + 1, (1,), dtype=torch.int32).item()
        y = torch.randint(0, height - size + 1, (1,), dtype=torch.int32).item()

        img[:, y : y + size, x : x + size] = 255
        return img


def create_dataloaders(
    path: str,
    model: nn.Module,
    crop_size: int,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cpu",
    eval_set: str = "test",
    max_samples: Optional[int] = None,
    cut_size: Optional[int] = None,
    hue_jitter: Optional[float] = None,
) -> DataLoaders:
    """
    Creates data loaders for the train and validation folds from the
    dataset stored within the provided directory.

    Args:
        path (Path): The path to the directory containing the dataset.
        model (Module): The PyTorch model used for training / inference.
        crop_size (int): Size of the square tile randomly extracted from each
            image.
        batch_size (int, optional): Minibatch size. Defaults to 8.
        num_workers (int, optional): Number of parallel worker threads.
            Defaults to 0.
        device (str, optional): Device to use for image transformations.
            Defaults to CPU.
        eval_set (str, optional): Subset of the dataset to use for evaluation.
            Defaults to "test".
        max_samples (int, optional): Maximum number of samples to use.
            No limit by default.
        cut_size (int, optional): Size of random square cuts used for image
            augmentation. Disabled by default.
        hue_jitter (float, optional): The magnitude of random hue jitter applied
            images. Disabled by default.

    Returns:
        DataLoaders: A dataclass containing the data loaders for the train
        and validation folds.
    """
    ds = _silent_load(path)

    normalize = model.image_normalization_layer()
    transform_train_ops = [
        PadImage(crop_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(180, fill=255),
        v2.RandomCrop(crop_size, fill=255),
    ]
    if cut_size is not None:
        transform_train_ops.append(RandomCut(cut_size))
    if hue_jitter is not None:
        transform_train_ops.append(v2.ColorJitter(hue=hue_jitter))
    transform_train_ops += [v2.ToDtype(torch.float32), normalize]
    transform_train = v2.Compose(transform_train_ops)

    transform_eval = v2.Compose(
        [
            PadImage(crop_size),
            FocusCrop(crop_size),
            v2.ToDtype(torch.float32),
            normalize,
        ]
    )

    dev = torch.device(device)
    tile_side = _compute_tile_size(model)

    train = ds["train"]
    train.set_transform(
        partial(_transform, transform=transform_train, tile_side=tile_side, device=dev)
    )
    if max_samples is not None and len(train) > max_samples:
        train = train.select(range(max_samples))

    eval = ds[eval_set]
    eval.set_transform(
        partial(_transform, transform=transform_eval, tile_side=tile_side, device=dev)
    )
    if max_samples is not None and len(eval) > max_samples:
        eval = eval.select(range(max_samples))

    persistent = True if num_workers > 0 else False
    return DataLoaders(
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent,
        ),
        DataLoader(
            eval,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent,
        ),
    )


def _silent_load(path: str) -> Dataset:
    progress_enabled = not are_progress_bars_disabled()
    if progress_enabled:
        disable_progress_bars()

    try:
        return load_dataset("imagefolder", data_dir=path)
    finally:
        if progress_enabled:
            enable_progress_bars()


def _compute_tile_size(model: BaseModule) -> int:
    _, w, h = model.image_size()
    assert w == h, "tiles should be squares, but the model supports rectangular images"
    return w


def _transform(
    batch: Dict, transform: v2.Transform, tile_side: int, device: torch.device
) -> Dict:
    out = _transform_metadata(batch)

    to_img = v2.ToImage()
    init = [to_img(img).to(device) for img in batch["image"]]

    out["tiles"] = torch.cat(
        [_image_tiles(transform(img), tile_side).unsqueeze(0) for img in init]
    )

    return out


def _transform_metadata(batch: Dict) -> Dict:
    fields = ["concentr", "has_age", "age", "order", "clade", "country", "has_coords"]
    out = {f: torch.tensor(batch[f]).unsqueeze(1) for f in fields}

    out["barcode"] = batch["barcode"]

    lat = torch.tensor(batch["latitude"]).unsqueeze(1)
    lon = torch.tensor(batch["longitude"]).unsqueeze(1)
    out["coords"] = torch.cat([lat, lon], dim=1)

    return out


def _image_tiles(image: torch.tensor, tile_side: int) -> torch.tensor:
    """Split image into square tiles of the given size."""
    side = image.size(1)
    assert side == image.size(2)
    assert side % tile_side == 0

    n = side // tile_side
    rows = []
    for i in range(n):
        x = i * tile_side

        cols = []
        for j in range(n):
            y = j * tile_side
            cols.append(image[:, y : y + tile_side, x : x + tile_side])

        rows.append(torch.stack(cols, dim=0))

    return torch.stack(rows)
