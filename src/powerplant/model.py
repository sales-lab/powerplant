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

from timm import create_model
from timm.data import resolve_data_config
from torch import nn
from torchvision.transforms import v2
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


class RegressionModel(BaseModule):
    """
    Regression model for herbarium images.
    """

    def __init__(
        self,
        order_num: int,
        clade_num: int,
        country_num: int,
        use_coords: bool,
        backend_name: str,
        embed_dim: int = 512,
        layer_num: int = 3,
        head_num: int = 8,
        feedforward_dim: int = 1024,
        transformer_dropout: float = 0.1,
        regr_dim: int = 256,
    ) -> None:
        """
        Create a regression model for herbarium images paired with metadata.

        Args:
            order_num (int): The number of taxonomic orders.
            clade_num (int): The number of taxonomic clades.
            country_num (int): The number of countries of origin.
            use_coords (bool): Whether to use the geographical coordinates.
            backend_name (str): The name of the backend model for feature
                extraction from the input images.
            embed_dim (int, optional): The size of the embedding used internally
                by the model. Defaults to 512.
            layer_num (int, optional): Number of transformer layers. Defaults
                to 3.
            head_num (int, optional): Number of transformer heads. Defaults to
                8.
            feedforward_dim (int, optional): Size of the feed-forward layer of
                the transformer stack. Defaults to 1024.
            transformer_dropout (float, optional): Dropout level for the
                transformer stack. Defaults to 0.1.
            regr_dim (int, optional): Size of the linear layer within the
                regression head. Defaults to 256.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        super(RegressionModel, self).__init__()
        self.d = embed_dim

        self.img_enc = ImageEncoder(backend_name, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d))

        self.has_age_embed = nn.Embedding(2, self.d)
        self.age_embed = nn.Linear(1, self.d)
        self.ord_embed = nn.Embedding(order_num, self.d)
        self.cla_embed = nn.Embedding(clade_num, self.d)
        self.ctr_embed = nn.Embedding(country_num, self.d)
        self.has_coords_embed = nn.Embedding(2, self.d) if use_coords else None
        self.crd_embed = nn.Linear(2, self.d) if use_coords else None

        encoder = nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=head_num,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=layer_num)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, regr_dim),
            nn.ReLU(),
            nn.Linear(regr_dim, 1),
            nn.Sigmoid(),
        )

    def image_size(self) -> Tuple[int, int, int]:
        """
        Returns the size of the input images expected by the model.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of channels,
               width and height of the image.
        """
        return self.img_enc.image_size()

    def image_normalization_layer(self) -> nn.Module:
        """
        Creates a normalization layer appropriate for this model.

        Returns:
            An instance of a torchvision normalization module.
        """
        return self.img_enc.image_normalization_layer()

    def backend_num_features(self) -> int:
        """
        Query the features provided by the backend model.

        Returns:
            The number of features, as an integer.
        """
        return self.img_backend.num_features

    def forward(
        self,
        tiles=None,
        has_age=None,
        age=None,
        order=None,
        clade=None,
        country=None,
        has_coords=None,
        coords=None,
        **kwargs,
    ):
        """
        Forward pass of the network.
        """
        cls_tok = self.cls_token.expand(tiles.size(0), 1, -1)
        has_age_tok = self.has_age_embed(has_age)
        age_tok = self.age_embed(age).unsqueeze(1)
        ord_tok = self.ord_embed(order)
        cla_tok = self.cla_embed(clade)
        ctr_tok = self.ctr_embed(country)
        toks = [cls_tok, has_age_tok, age_tok, ord_tok, cla_tok, ctr_tok]

        if self.crd_embed:
            has_coords_tok = self.has_coords_embed(has_coords)
            crd_tok = self.crd_embed(coords).unsqueeze(1)
            toks += [has_coords_tok, crd_tok]

        toks.append(self.img_enc(tiles))

        x = torch.cat(toks, dim=1)
        x = self.transformer(x)
        x = self.regression_head(x[:, 0])
        return x

    def save_image_encoder_state(self, filename: str) -> None:
        """
        Save the image encoder state to a file.
        """
        self.img_enc.save_state(filename)


class ImageEncoder(BaseModule):
    def __init__(
        self,
        backend_name: str,
        embed_dim: int = 512,
    ) -> None:
        """
        Create an image encoder based on an arbitry backend.

        Args:
            backend_name (str): The name of the backend model for feature
                extraction from the input images.
            embed_dim (int, optional): The size of the embedding used internally
                by the model. Defaults to 512.
        """
        super(ImageEncoder, self).__init__()
        self.d = embed_dim

        self.img_backend = _load_backend_model(backend_name)
        self.backend_config = resolve_data_config(
            self.img_backend.pretrained_cfg, model=backend_name
        )

        fn = self.img_backend.num_features
        self.img_norm = nn.LayerNorm(fn)

        if fn == self.d:
            self.img_proj = nn.Identity(fn)
        else:
            self.img_proj = nn.Linear(fn, self.d)

    def image_size(self) -> Tuple[int, int, int]:
        """
        Returns the size of the input images expected by the model.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of channels,
               width and height of the image.
        """
        return self.backend_config["input_size"]

    def image_normalization_layer(self) -> nn.Module:
        """
        Creates a normalization layer appropriate for this model.

        Returns:
            An instance of a torchvision normalization module.
        """
        cfg = self.backend_config
        return v2.Normalize(mean=cfg["mean"], std=cfg["std"])

    def backend_num_features(self) -> int:
        """
        Query the features provided by the backend model.

        Returns:
            The number of features, as an integer.
        """
        return self.img_backend.num_features

    def forward(self, tiles=None, **kwargs) -> torch.tensor:
        """
        Forward pass of the network.
        """
        b = tiles.size(0)

        x = tiles.view(-1, *self.image_size())
        x = self.img_backend.forward(x)
        x = self.img_norm(x)
        x = self.img_proj(x)
        x = x.view(b, -1, self.d)

        return x


def _load_backend_model(name: str) -> nn.Module:
    model = create_model(name, pretrained=True, num_classes=0)

    with torch.no_grad():
        batch = torch.zeros((1, 3, 224, 224))
        out = model(batch)
        assert len(out.size()) == 2
        model.num_features = out.size(1)

    return model
