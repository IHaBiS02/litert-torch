# Copyright 2024 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for SDXL.

Reuses get_time_embedding, get_alphas_cumprod, move_channel, rescale from SD 1.5.
Adds SDXL-specific get_add_time_ids() and encode_add_time_ids().
"""

import numpy as np
import torch

# Reuse utilities from SD 1.5
from litert_torch.generative.examples.stable_diffusion.util import (
    get_alphas_cumprod,
    get_file_path,
    get_time_embedding,
    move_channel,
    rescale,
)


def get_add_time_ids(
    original_size=(1024, 1024),
    crop_coords=(0, 0),
    target_size=(1024, 1024),
):
  """Create add_time_ids tensor for SDXL additional conditioning.

  Args:
    original_size: (height, width) of the original image.
    crop_coords: (top, left) crop coordinates.
    target_size: (height, width) of the target image.

  Returns:
    Tensor of shape (1, 6) with the 6 conditioning values.
  """
  add_time_ids = list(original_size) + list(crop_coords) + list(target_size)
  return torch.tensor([add_time_ids], dtype=torch.float32)


def encode_add_time_ids(add_time_ids, embedding_dim=256):
  """Encode add_time_ids using sinusoidal embeddings, same as timestep encoding.

  Args:
    add_time_ids: Tensor of shape (batch_size, 6).
    embedding_dim: Dimension of each sinusoidal embedding. Default 256.

  Returns:
    Tensor of shape (batch_size, 6 * embedding_dim) = (batch_size, 1536).
  """
  half_dim = embedding_dim // 2
  freqs = torch.pow(
      10000, -torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
  )
  embeddings = []
  for i in range(add_time_ids.shape[1]):
    val = add_time_ids[:, i:i+1]
    x = val * freqs[None]
    emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    embeddings.append(emb)
  return torch.cat(embeddings, dim=-1)
