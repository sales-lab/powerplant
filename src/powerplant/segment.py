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
from glob import glob
from os.path import abspath, basename, exists, join
from PIL import Image
from shlex import quote
from subprocess import check_call
from tqdm import tqdm

import cv2
import logging
import numpy as np
import warnings


logger = logging.getLogger(__name__)


class MissingMask(Exception):
    "A segmentation mask for an image could not be found."


@dataclass
class Resolution:
    height: int
    label: str
    dilation: int


def generate_masks(input_dir: str, output_dir: str) -> None:
    """
    Generate segmentation masks to isolate plant material and remove
    annotations, labels, stamps, and envelopes.
    """
    inp = quote(abspath(input_dir))
    outp = quote(abspath(output_dir))

    logging.info("Starting PaddleSeg segmentation.")

    cmd = "source .venv/bin/activate && python deploy/python/infer.py " \
          "--config ../../checkpoints/segment/deploy.yaml " \
          f"--image_path {inp} --save_dir {outp} --cpu_threads $(nproc)"
    check_call(cmd, cwd="vendor/PaddleSeg", shell=True)

    mask_num = len(glob(join(output_dir, "*.png")))
    logging.info(f"Generated {mask_num} image masks.")


def mask_images(input_dir: str, mask_dir: str, output_dir: str, height: int,
                min_comp_area: int, nonempty_min_ratio: float) -> None:
    """
    Apply segmentation masks to herbarium images.
    """
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)

    images = sorted(glob(join(input_dir, "*.jpg")))

    for img in tqdm(images, desc="Image masking", total=len(images),
                    leave=False):
        mask = join(mask_dir, basename(img).replace(".jpg", ".png"))
        if not exists(mask):
            raise MissingMask(f"Could not find mask file: {mask}")

        _mask_image(img, mask, output_dir, height, min_comp_area,
                    nonempty_min_ratio)

    logger.info(f"Masked {len(images)} images.")


def _mask_image(image_path: str, mask_path: str, dest_dir: str, height: int,
                min_comp_area: int, nonempty_min_ratio: float) -> None:
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = _clean_mask(mask, min_comp_area)

    img = _resize_image(image, height)
    msk = _resize_image(mask, height)
    img = _apply_mask(img, msk)
    img = _clean_image(img, nonempty_min_ratio)

    if img:
        target = join(dest_dir, basename(image_path).replace(".jpg", ".png"))
        img.save(target)


def _clean_mask(mask, min_comp_area):
    mask = mask.point(lambda x: 1 if x > 0 else 0, "1")
    mask = mask.convert("L")

    mask2 = np.array(mask, dtype=np.uint8)
    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(mask2, connectivity=8)

    mask3 = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_comp_area:
            mask3[labels == i] = 255

    mask = np.logical_and(mask, mask3)
    return Image.fromarray(mask)


def _resize_image(img, target_height):
    w, h = img.size

    if w > h:
        img = img.transpose(Image.Transpose.ROTATE_90)
        w, h = img.size

    if h > target_height:
        scale = h / target_height
        w = int(round(w / scale))
        img = img.resize((w, target_height))

    return img


def _apply_mask(image, mask):
    mask = mask.point(lambda x: 1 if x > 0 else 0, "1")

    mask = mask.convert("L")
    if image.size != mask.size:
        mask = mask.resize((image.size[0], image.size[1]))

    mask = np.array(mask)[:, :, np.newaxis]
    out = np.where(mask > 0, np.array(image), np.ones_like(image) * 255)
    return Image.fromarray(out)


def _clean_image(image, nonempty_min_frac):
    orig_size = image.size

    image = _crop_image(image)
    if image is None:
        return None

    image = _filter_almost_empty(image, orig_size, nonempty_min_frac)
    if image is None:
        return None

    return image


def _crop_image(image):
    mask = image.convert("L")
    mask = mask.point(lambda x: 255 if x < 255 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    if (x2 - x1) % 2 != 0:
        if x2 < image.size[0]:
            x2 += 1
        else:
            x1 -= 1

    if (y2 - y1) % 2 != 0:
        if y2 < image.size[1]:
            y2 += 1
        else:
            y1 -= 1

    image = image.crop((x1, y1, x2, y2))
    if image.size[0] > image.size[1]:
        image = image.transpose(Image.Transpose.ROTATE_90)

    return image


def _filter_almost_empty(image, orig_size, min_nonempty_ratio):
    mask = image.convert("L")
    non_white_pixels = mask.size[0] * mask.size[1] - mask.histogram()[-1]
    total_pixels = orig_size[0] * orig_size[1]
    ratio = non_white_pixels / total_pixels
    return image if ratio >= min_nonempty_ratio else None
