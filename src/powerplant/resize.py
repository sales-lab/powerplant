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
from glob import glob
from multiprocessing import Pool
from os.path import basename, join
from PIL import Image
from tqdm import tqdm

import logging
import warnings


logger = logging.getLogger(__name__)


@dataclass
class Summary:
    flipped: bool
    low_size: bool


class NoImagesFound(Exception):
    "No input images found."


def resize_images(input_dir: str, output_dir: str, target_size: int) -> None:
    """
    Resize images so that their largest side is at most of the given size.
    """
    images = sorted(glob(join(input_dir, "*.jpg")))
    if len(images) == 0:
        raise NoImagesFound()

    num = len(images)
    logger.info(f"Found {num} source images.")

    with Pool() as p:
        it = p.imap(partial(_process, dest=output_dir, size=target_size),
                    images)
        flipped = 0
        low_size = 0

        for s in tqdm(it, desc="Image resizing", total=num, leave=False):
            if s.flipped:
                flipped += 1
            if s.low_size:
                low_size += 1

    logger.info(f"Resized {num} images.")
    if flipped > 0:
        logger.info(f"{flipped} images were rotated.")
    if low_size > 0:
        logger.info(f"{low_size} were not resized (too small).")


def _process(path: str, dest: str, size: int) -> Summary:
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)

    s = Summary(False, False)

    image = Image.open(path)
    w, h = image.size

    if w > h:
        image = image.transpose(Image.Transpose.ROTATE_90)
        w, h = image.size
        s.flipped = True

    if h < size:
        s.low_size = True
    elif h > size:
        scale = h / size
        w = int(round(w / scale))
        h = size
        image = image.resize((w, h))

    dest = join(dest, basename(path).replace(".jpg", ".png"))
    image.save(dest)

    return s
