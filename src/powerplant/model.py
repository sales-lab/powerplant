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

from torch import nn
from typing import Tuple

import torch


class BaseModule(nn.Module):
    def image_size(self) -> Tuple[int, int, int]:
        """
        Returns the size of the input images expected by the model.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of channels,
               width and height of the image.
        """
        raise NotImplementedError()

    def save_state(self, filename: str) -> None:
        """
        Save the model state to a file.
        """
        torch.save(self.state_dict(), filename)

    def load_state(self, filename: str) -> None:
        """
        Load the model state from a file.
        """
        state = torch.load(
            filename, weights_only=True, map_location=torch.device("cpu")
        )
        self.load_state_dict(state)
