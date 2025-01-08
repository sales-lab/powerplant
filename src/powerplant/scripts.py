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

from powerplant.resize import NoImagesFound, resize_images
from powerplant.segment import MissingMask, generate_masks, mask_images

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
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
