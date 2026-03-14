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

"""VAE Decoder for SDXL.

Reuses the same Decoder class and TENSOR_NAMES as SD 1.5.
The only difference is scaling_factor=0.13025 (vs 0.18215 for SD 1.5).
"""

import litert_torch
import litert_torch.generative.layers.builder as layers_builder
import litert_torch.generative.layers.model_config as layers_cfg
from litert_torch.generative.layers.unet import blocks_2d
import litert_torch.generative.layers.unet.model_config as unet_cfg
from litert_torch.generative.utilities import stable_diffusion_loader

# Reuse the same Decoder class from SD 1.5
from litert_torch.generative.examples.stable_diffusion.decoder import Decoder
from litert_torch.generative.examples.stable_diffusion.decoder import TENSOR_NAMES


def get_model_config(device_type: str = "cpu") -> unet_cfg.AutoEncoderConfig:
  """Get configs for the VAE Decoder of SDXL.

  Same architecture as SD 1.5 but with scaling_factor=0.13025.
  """
  in_channels = 3
  latent_channels = 4
  out_channels = 3
  block_out_channels = [128, 256, 512, 512]
  scaling_factor = 0.13025
  layers_per_block = 3

  enable_hlfb = False
  if device_type == "gpu":
    litert_torch.config.enable_group_norm_composite = True
    enable_hlfb = True

  norm_config = layers_cfg.NormalizationConfig(
      layers_cfg.NormalizationType.GROUP_NORM,
      group_num=32,
      enable_hlfb=enable_hlfb,
  )

  att_config = unet_cfg.AttentionBlock2DConfig(
      dim=block_out_channels[-1],
      normalization_config=norm_config,
      attention_config=layers_cfg.AttentionConfig(
          num_heads=1,
          head_dim=block_out_channels[-1],
          num_query_groups=1,
          qkv_use_bias=True,
          output_proj_use_bias=True,
          enable_kv_cache=False,
          qkv_transpose_before_split=True,
          qkv_fused_interleaved=False,
          rotary_base=0,
          rotary_percentage=0.0,
      ),
      enable_hlfb=enable_hlfb,
  )

  mid_block_config = unet_cfg.MidBlock2DConfig(
      in_channels=block_out_channels[-1],
      normalization_config=norm_config,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      num_layers=1,
      attention_block_config=att_config,
  )

  config = unet_cfg.AutoEncoderConfig(
      in_channels=in_channels,
      latent_channels=latent_channels,
      out_channels=out_channels,
      activation_config=layers_cfg.ActivationConfig(
          layers_cfg.ActivationType.SILU
      ),
      block_out_channels=block_out_channels,
      scaling_factor=scaling_factor,
      layers_per_block=layers_per_block,
      normalization_config=norm_config,
      mid_block_config=mid_block_config,
  )
  return config
