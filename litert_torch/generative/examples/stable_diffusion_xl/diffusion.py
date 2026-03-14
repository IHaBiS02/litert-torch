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

"""SDXL Diffusion (UNet) model.

Key differences from SD 1.5:
  - 3 stages instead of 4: [320, 640, 1280]
  - Multi-layer transformer blocks: [0, 2, 10] layers per block
  - Attention heads via head_dim=64: [5, 10, 20] per stage
  - Additional conditioning: AddEmbedding(2816 -> 1280) from pooled text + time_ids
  - Cross-attention dim: 2048 (768+1280 from dual encoders)
  - Linear proj_in/proj_out in transformer blocks (not Conv2d)
"""

import litert_torch
import litert_torch.generative.layers.builder as layers_builder
import litert_torch.generative.layers.model_config as layers_cfg
from litert_torch.generative.layers.unet import blocks_2d
import litert_torch.generative.layers.unet.model_config as unet_cfg
from litert_torch.generative.utilities import stable_diffusion_loader
import torch
from torch import nn

from typing import Dict, List, Tuple


def build_attention_config(
    num_heads,
    dim,
    num_query_groups,
    rotary_base=0,
    rotary_percentage=0.0,
    qkv_transpose_before_split=True,
    qkv_use_bias=False,
    output_proj_use_bias=True,
    enable_kv_cache=False,
    qkv_fused_interleaved=False,
):
  return layers_cfg.AttentionConfig(
      num_heads=num_heads,
      head_dim=dim // num_heads,
      num_query_groups=num_query_groups,
      rotary_base=rotary_base,
      rotary_percentage=rotary_percentage,
      qkv_transpose_before_split=qkv_transpose_before_split,
      qkv_use_bias=qkv_use_bias,
      output_proj_use_bias=output_proj_use_bias,
      enable_kv_cache=enable_kv_cache,
      qkv_fused_interleaved=qkv_fused_interleaved,
  )


class TimeEmbedding(nn.Module):

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.w1 = nn.Linear(in_dim, out_dim)
    self.w2 = nn.Linear(out_dim, out_dim)
    self.act = layers_builder.get_activation(
        layers_cfg.ActivationConfig(layers_cfg.ActivationType.SILU)
    )

  def forward(self, x: torch.Tensor):
    return self.w2(self.act(self.w1(x)))


class AddEmbedding(nn.Module):
  """Additional embedding module for SDXL.

  Maps concatenated [pooled_text_embed(1280), time_ids_embed(1536)] = 2816
  to time_embedding_blocks_dim (1280).
  """

  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.w1 = nn.Linear(in_dim, out_dim)
    self.w2 = nn.Linear(out_dim, out_dim)
    self.act = layers_builder.get_activation(
        layers_cfg.ActivationConfig(layers_cfg.ActivationType.SILU)
    )

  def forward(self, x: torch.Tensor):
    return self.w2(self.act(self.w1(x)))


# SDXL block configuration
# Block 0: 320ch, no transformer (DownBlock2D)
# Block 1: 640ch, 2 transformer layers per block, 10 heads
# Block 2: 1280ch, 10 transformer layers per block, 20 heads
_BLOCK_OUT_CHANNELS = [320, 640, 1280]
_TRANSFORMER_LAYERS_PER_BLOCK = [0, 2, 10]
_HEAD_DIM = 64
_CROSS_ATTENTION_DIM = 2048
_TIME_EMBEDDING_DIM = 320
_TIME_EMBEDDING_BLOCKS_DIM = 1280
_ADD_EMBEDDING_IN_DIM = 2816  # 1280 (pooled) + 1536 (6*256 time_ids)
_LAYERS_PER_BLOCK = 2
_MID_BLOCK_TRANSFORMER_LAYERS = 10


def _build_transformer_block_config(
    out_channel,
    num_heads,
    transformer_batch_size,
    transformer_norm_config,
    transformer_pre_conv_norm_config,
    enable_hlfb,
):
  """Build TransformerBlock2DConfig for SDXL blocks."""
  return unet_cfg.TransformerBlock2DConfig(
      attention_block_config=unet_cfg.AttentionBlock2DConfig(
          dim=out_channel,
          attention_batch_size=transformer_batch_size,
          normalization_config=transformer_norm_config,
          attention_config=build_attention_config(
              num_heads=num_heads,
              dim=out_channel,
              num_query_groups=num_heads,
          ),
          enable_hlfb=enable_hlfb,
      ),
      cross_attention_block_config=unet_cfg.CrossAttentionBlock2DConfig(
          query_dim=out_channel,
          cross_dim=_CROSS_ATTENTION_DIM,
          hidden_dim=out_channel,
          output_dim=out_channel,
          attention_batch_size=transformer_batch_size,
          normalization_config=transformer_norm_config,
          attention_config=build_attention_config(
              num_heads=num_heads,
              dim=out_channel,
              num_query_groups=num_heads,
          ),
          enable_hlfb=enable_hlfb,
      ),
      pre_conv_normalization_config=transformer_pre_conv_norm_config,
      feed_forward_block_config=unet_cfg.FeedForwardBlock2DConfig(
          dim=out_channel,
          hidden_dim=out_channel * 4,
          normalization_config=transformer_norm_config,
          activation_config=layers_cfg.ActivationConfig(
              type=layers_cfg.ActivationType.GE_GLU,
              dim_in=out_channel,
              dim_out=out_channel * 4,
          ),
          use_bias=True,
      ),
  )


class SDXLDiffusion(nn.Module):
  """The SDXL Diffusion (UNet) model.

  3 stages with [320, 640, 1280] channels.
  Multi-layer transformers: [0, 2, 10] layers per block.
  Additional conditioning via AddEmbedding.
  """

  def __init__(
      self,
      batch_size: int = 2,
      device_type: str = "cpu",
  ):
    super().__init__()

    enable_hlfb = False
    if device_type == "gpu":
      litert_torch.config.enable_group_norm_composite = True
      enable_hlfb = True

    self.layers_per_block = _LAYERS_PER_BLOCK
    block_out_channels = _BLOCK_OUT_CHANNELS

    residual_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM,
        group_num=32,
        enable_hlfb=enable_hlfb,
    )
    residual_activation_type = layers_cfg.ActivationType.SILU
    transformer_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.LAYER_NORM,
        enable_hlfb=enable_hlfb,
    )
    transformer_pre_conv_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM,
        epsilon=1e-6,
        group_num=32,
        enable_hlfb=enable_hlfb,
    )

    # Time embedding
    self.time_embedding = TimeEmbedding(
        _TIME_EMBEDDING_DIM, _TIME_EMBEDDING_BLOCKS_DIM
    )

    # Additional embedding (SDXL-specific)
    self.add_embedding = AddEmbedding(
        _ADD_EMBEDDING_IN_DIM, _TIME_EMBEDDING_BLOCKS_DIM
    )

    # Conv in
    self.conv_in = nn.Conv2d(4, block_out_channels[0], kernel_size=3, padding=1)

    # Down encoders (3 blocks)
    down_encoders = []
    output_channel = block_out_channels[0]
    for i, block_out_channel in enumerate(block_out_channels):
      input_channel = output_channel
      output_channel = block_out_channel
      not_final_block = i < len(block_out_channels) - 1
      num_heads = output_channel // _HEAD_DIM
      num_tl = _TRANSFORMER_LAYERS_PER_BLOCK[i]
      has_transformer = num_tl > 0

      transformer_config = None
      if has_transformer:
        transformer_config = _build_transformer_block_config(
            output_channel, num_heads, batch_size,
            transformer_norm_config, transformer_pre_conv_norm_config,
            enable_hlfb,
        )

      down_encoders.append(
          blocks_2d.DownEncoderBlock2D(
              unet_cfg.DownEncoderBlock2DConfig(
                  in_channels=input_channel,
                  out_channels=output_channel,
                  normalization_config=residual_norm_config,
                  activation_config=layers_cfg.ActivationConfig(
                      residual_activation_type
                  ),
                  num_layers=_LAYERS_PER_BLOCK,
                  padding=1,
                  time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
                  add_downsample=not_final_block,
                  sampling_config=unet_cfg.DownSamplingConfig(
                      mode=unet_cfg.SamplingType.CONVOLUTION,
                      in_channels=output_channel,
                      out_channels=output_channel,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                  ) if not_final_block else None,
                  transformer_block_config=transformer_config,
                  num_transformer_layers=num_tl if has_transformer else 1,
              )
          )
      )
    self.down_encoders = nn.ModuleList(down_encoders)

    # Mid block
    mid_block_channels = block_out_channels[-1]
    mid_num_heads = mid_block_channels // _HEAD_DIM
    self.mid_block = blocks_2d.MidBlock2D(
        unet_cfg.MidBlock2DConfig(
            in_channels=mid_block_channels,
            normalization_config=residual_norm_config,
            activation_config=layers_cfg.ActivationConfig(
                residual_activation_type
            ),
            num_layers=1,
            time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
            transformer_block_config=_build_transformer_block_config(
                mid_block_channels, mid_num_heads, batch_size,
                transformer_norm_config, transformer_pre_conv_norm_config,
                enable_hlfb,
            ),
            num_transformer_layers=_MID_BLOCK_TRANSFORMER_LAYERS,
        )
    )

    # Up decoders (3 blocks, mirror of down)
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_transformer_layers = list(reversed(_TRANSFORMER_LAYERS_PER_BLOCK))
    up_decoder_layers_per_block = _LAYERS_PER_BLOCK + 1
    up_decoders = []
    output_channel = reversed_block_out_channels[0]
    for i, block_out_channel in enumerate(reversed_block_out_channels):
      prev_out_channel = output_channel
      output_channel = block_out_channel
      input_channel = reversed_block_out_channels[
          min(i + 1, len(reversed_block_out_channels) - 1)
      ]
      not_final_block = i < len(reversed_block_out_channels) - 1
      num_heads = output_channel // _HEAD_DIM
      num_tl = reversed_transformer_layers[i]
      has_transformer = num_tl > 0

      transformer_config = None
      if has_transformer:
        transformer_config = _build_transformer_block_config(
            output_channel, num_heads, batch_size,
            transformer_norm_config, transformer_pre_conv_norm_config,
            enable_hlfb,
        )

      up_decoders.append(
          blocks_2d.SkipUpDecoderBlock2D(
              unet_cfg.SkipUpDecoderBlock2DConfig(
                  in_channels=input_channel,
                  out_channels=output_channel,
                  prev_out_channels=prev_out_channel,
                  normalization_config=residual_norm_config,
                  activation_config=layers_cfg.ActivationConfig(
                      residual_activation_type
                  ),
                  num_layers=up_decoder_layers_per_block,
                  time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
                  add_upsample=not_final_block,
                  upsample_conv=True,
                  sampling_config=unet_cfg.UpSamplingConfig(
                      mode=unet_cfg.SamplingType.NEAREST,
                      scale_factor=2,
                  ),
                  transformer_block_config=transformer_config,
                  num_transformer_layers=num_tl if has_transformer else 1,
              )
          )
      )
    self.up_decoders = nn.ModuleList(up_decoders)

    # Final layers
    final_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM,
        group_num=32,
        enable_hlfb=enable_hlfb,
    )
    self.final_norm = layers_builder.build_norm(
        reversed_block_out_channels[-1], final_norm_config
    )
    self.final_act = layers_builder.get_activation(
        layers_cfg.ActivationConfig(layers_cfg.ActivationType.SILU)
    )
    self.conv_out = nn.Conv2d(
        reversed_block_out_channels[-1], 4, kernel_size=3, padding=1
    )

  @torch.inference_mode
  def forward(
      self,
      latents: torch.Tensor,
      context: torch.Tensor,
      time_emb: torch.Tensor,
      add_emb: torch.Tensor,
  ) -> torch.Tensor:
    """Forward function of SDXL diffusion model.

    Args:
        latents: Latent space tensor (batch, 4, H/8, W/8).
        context: Context tensor from dual encoders (batch, 77, 2048).
        time_emb: Time embedding tensor (1, 320).
        add_emb: Additional embedding (batch, 2816) from pooled text + time_ids.

    Returns:
        Predicted noise tensor.
    """
    time_emb = self.time_embedding(time_emb)
    add_emb = self.add_embedding(add_emb)
    time_emb = time_emb + add_emb

    x = self.conv_in(latents)
    skip_connection_tensors = [x]
    for encoder in self.down_encoders:
      x, hidden_states = encoder(
          x, time_emb, context, output_hidden_states=True
      )
      skip_connection_tensors.extend(hidden_states)
    x = self.mid_block(x, time_emb, context)
    for decoder in self.up_decoders:
      encoder_tensors = [
          skip_connection_tensors.pop()
          for _ in range(self.layers_per_block + 1)
      ]
      x = decoder(x, encoder_tensors, time_emb, context)
    x = self.final_norm(x)
    x = self.final_act(x)
    x = self.conv_out(x)
    return x


# ==============================================================================
# Tensor name mappings for SDXL checkpoint
# ==============================================================================

def _make_multi_layer_transformer_names(prefix, num_layers):
  """Create MultiLayerTransformerBlockTensorNames for SDXL blocks."""
  return stable_diffusion_loader.MultiLayerTransformerBlockTensorNames(
      pre_conv_norm=f"{prefix}.norm",
      proj_in=f"{prefix}.proj_in",
      self_attentions=[
          stable_diffusion_loader.AttentionBlockTensorNames(
              norm=f"{prefix}.transformer_blocks.{l}.norm1",
              q_proj=f"{prefix}.transformer_blocks.{l}.attn1.to_q",
              k_proj=f"{prefix}.transformer_blocks.{l}.attn1.to_k",
              v_proj=f"{prefix}.transformer_blocks.{l}.attn1.to_v",
              output_proj=f"{prefix}.transformer_blocks.{l}.attn1.to_out.0",
          )
          for l in range(num_layers)
      ],
      cross_attentions=[
          stable_diffusion_loader.CrossAttentionBlockTensorNames(
              norm=f"{prefix}.transformer_blocks.{l}.norm2",
              q_proj=f"{prefix}.transformer_blocks.{l}.attn2.to_q",
              k_proj=f"{prefix}.transformer_blocks.{l}.attn2.to_k",
              v_proj=f"{prefix}.transformer_blocks.{l}.attn2.to_v",
              output_proj=f"{prefix}.transformer_blocks.{l}.attn2.to_out.0",
          )
          for l in range(num_layers)
      ],
      feed_forwards=[
          stable_diffusion_loader.FeedForwardBlockTensorNames(
              norm=f"{prefix}.transformer_blocks.{l}.norm3",
              ge_glu=f"{prefix}.transformer_blocks.{l}.ff.net.0.proj",
              w2=f"{prefix}.transformer_blocks.{l}.ff.net.2",
          )
          for l in range(num_layers)
      ],
      proj_out=f"{prefix}.proj_out",
  )


# SDXL input block numbering:
# 0: conv_in
# 1,2: down_block_0 resnets (no transformer)
# 3: downsample_0
# 4,5: down_block_1 resnets + transformer (2 layers)
# 6: downsample_1
# 7,8: down_block_2 resnets + transformer (10 layers)
_INPUT_BLOCK_INDICES = [
    # Block 0: no transformer, indices 1,2
    [1, 2],
    # Block 1: 2 transformer layers, indices 4,5
    [4, 5],
    # Block 2: 10 transformer layers, indices 7,8
    [7, 8],
]
_DOWNSAMPLE_INDICES = [3, 6]  # For blocks 0 and 1

_down_encoder_blocks_tensor_names = []
for block_idx in range(3):
  indices = _INPUT_BLOCK_INDICES[block_idx]
  num_tl = _TRANSFORMER_LAYERS_PER_BLOCK[block_idx]
  has_transformer = num_tl > 0
  has_downsample = block_idx < 2

  residual_names = []
  for j, idx in enumerate(indices):
    residual_names.append(
        stable_diffusion_loader.ResidualBlockTensorNames(
            norm_1=f"model.diffusion_model.input_blocks.{idx}.0.in_layers.0",
            conv_1=f"model.diffusion_model.input_blocks.{idx}.0.in_layers.2",
            norm_2=f"model.diffusion_model.input_blocks.{idx}.0.out_layers.0",
            conv_2=f"model.diffusion_model.input_blocks.{idx}.0.out_layers.3",
            time_embedding=f"model.diffusion_model.input_blocks.{idx}.0.emb_layers.1",
            residual_layer=f"model.diffusion_model.input_blocks.{idx}.0.skip_connection"
            if (block_idx > 0 and j == 0)
            else None,
        )
    )

  transformer_names = None
  if has_transformer:
    transformer_names = [
        _make_multi_layer_transformer_names(
            f"model.diffusion_model.input_blocks.{idx}.1", num_tl
        )
        for idx in indices
    ]

  _down_encoder_blocks_tensor_names.append(
      stable_diffusion_loader.DownEncoderBlockTensorNames(
          residual_block_tensor_names=residual_names,
          transformer_block_tensor_names=transformer_names,
          downsample_conv=(
              f"model.diffusion_model.input_blocks.{_DOWNSAMPLE_INDICES[block_idx]}.0.op"
              if has_downsample
              else None
          ),
      )
  )


# Mid block
_mid_block_tensor_names = stable_diffusion_loader.MidBlockTensorNames(
    residual_block_tensor_names=[
        stable_diffusion_loader.ResidualBlockTensorNames(
            norm_1=f"model.diffusion_model.middle_block.{i}.in_layers.0",
            conv_1=f"model.diffusion_model.middle_block.{i}.in_layers.2",
            norm_2=f"model.diffusion_model.middle_block.{i}.out_layers.0",
            conv_2=f"model.diffusion_model.middle_block.{i}.out_layers.3",
            time_embedding=f"model.diffusion_model.middle_block.{i}.emb_layers.1",
        )
        for i in [0, 2]
    ],
    transformer_block_tensor_names=[
        _make_multi_layer_transformer_names(
            "model.diffusion_model.middle_block.1",
            _MID_BLOCK_TRANSFORMER_LAYERS,
        )
    ],
)


# SDXL output block numbering:
# Up block 0 (1280ch, 10 transformer layers): output_blocks 0,1,2
# Up block 1 (640ch, 2 transformer layers): output_blocks 3,4,5
# Up block 2 (320ch, no transformer): output_blocks 6,7,8
_OUTPUT_BLOCK_INDICES = [
    [0, 1, 2],  # Up block 0: 10 transformer layers
    [3, 4, 5],  # Up block 1: 2 transformer layers
    [6, 7, 8],  # Up block 2: no transformer
]
_reversed_transformer_layers = list(reversed(_TRANSFORMER_LAYERS_PER_BLOCK))

_up_decoder_blocks_tensor_names = []
for block_idx in range(3):
  indices = _OUTPUT_BLOCK_INDICES[block_idx]
  num_tl = _reversed_transformer_layers[block_idx]
  has_transformer = num_tl > 0

  residual_names = [
      stable_diffusion_loader.ResidualBlockTensorNames(
          norm_1=f"model.diffusion_model.output_blocks.{idx}.0.in_layers.0",
          conv_1=f"model.diffusion_model.output_blocks.{idx}.0.in_layers.2",
          norm_2=f"model.diffusion_model.output_blocks.{idx}.0.out_layers.0",
          conv_2=f"model.diffusion_model.output_blocks.{idx}.0.out_layers.3",
          time_embedding=f"model.diffusion_model.output_blocks.{idx}.0.emb_layers.1",
          residual_layer=f"model.diffusion_model.output_blocks.{idx}.0.skip_connection",
      )
      for idx in indices
  ]

  transformer_names = None
  if has_transformer:
    transformer_names = [
        _make_multi_layer_transformer_names(
            f"model.diffusion_model.output_blocks.{idx}.1", num_tl
        )
        for idx in indices
    ]

  # Upsample: blocks 0 and 1 have upsample at the last output block position
  upsample_name = None
  if block_idx < 2:
    last_idx = indices[-1]
    upsample_sub_idx = 2 if has_transformer else 1
    upsample_name = f"model.diffusion_model.output_blocks.{last_idx}.{upsample_sub_idx}.conv"

  _up_decoder_blocks_tensor_names.append(
      stable_diffusion_loader.SkipUpDecoderBlockTensorNames(
          residual_block_tensor_names=residual_names,
          transformer_block_tensor_names=transformer_names,
          upsample_conv=upsample_name,
      )
  )


# ==============================================================================
# SDXL Diffusion Model Loader
# ==============================================================================

class SDXLDiffusionModelLoader(stable_diffusion_loader.BaseLoader):
  """Custom loader for SDXL diffusion model."""

  def __init__(self, file_name: str):
    self._file_name = file_name
    self._loader = self._get_loader()

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    state = self._loader(self._file_name)
    converted_state = dict()

    # Time embedding
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.time_embed.0",
        converted_state, "time_embedding.w1",
    )
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.time_embed.2",
        converted_state, "time_embedding.w2",
    )

    # Add embedding (SDXL-specific: label_emb)
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.label_emb.0.0",
        converted_state, "add_embedding.w1",
    )
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.label_emb.0.2",
        converted_state, "add_embedding.w2",
    )

    # Conv in/out, final norm
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.input_blocks.0.0",
        converted_state, "conv_in",
    )
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.out.2",
        converted_state, "conv_out",
    )
    stable_diffusion_loader._map_to_converted_state(
        state, "model.diffusion_model.out.0",
        converted_state, "final_norm",
    )

    # Down encoder blocks
    for i in range(3):
      self._map_sdxl_down_encoder_block(
          state, converted_state, i,
          _down_encoder_blocks_tensor_names[i],
      )

    # Mid block
    self._map_sdxl_mid_block(state, converted_state)

    # Up decoder blocks
    for i in range(3):
      self._map_sdxl_up_decoder_block(
          state, converted_state, i,
          _up_decoder_blocks_tensor_names[i],
      )

    if strict and state:
      raise ValueError(
          f"Failed to map all tensors. Remaining: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)

  def _map_sdxl_down_encoder_block(
      self,
      state,
      converted_state,
      block_idx,
      tensor_names,
  ):
    num_tl = _TRANSFORMER_LAYERS_PER_BLOCK[block_idx]
    has_transformer = num_tl > 0
    out_channel = _BLOCK_OUT_CHANNELS[block_idx]
    in_channel = _BLOCK_OUT_CHANNELS[block_idx - 1] if block_idx > 0 else out_channel

    residual_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM, group_num=32,
    )
    residual_activation_config = layers_cfg.ActivationConfig(
        layers_cfg.ActivationType.SILU
    )
    transformer_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.LAYER_NORM,
    )

    for j in range(_LAYERS_PER_BLOCK):
      input_channels = in_channel if j == 0 else out_channel
      self._map_residual_block(
          state, converted_state,
          tensor_names.residual_block_tensor_names[j],
          f"down_encoders.{block_idx}.resnets.{j}",
          unet_cfg.ResidualBlock2DConfig(
              in_channels=input_channels,
              hidden_channels=out_channel,
              out_channels=out_channel,
              time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
              normalization_config=residual_norm_config,
              activation_config=residual_activation_config,
          ),
      )
      if has_transformer:
        num_heads = out_channel // _HEAD_DIM
        transformer_config = _build_transformer_block_config(
            out_channel, num_heads, 1,
            transformer_norm_config,
            layers_cfg.NormalizationConfig(
                layers_cfg.NormalizationType.GROUP_NORM,
                epsilon=1e-6, group_num=32,
            ),
            False,
        )
        self._map_multi_layer_transformer_block(
            state, converted_state,
            tensor_names.transformer_block_tensor_names[j],
            f"down_encoders.{block_idx}.transformers.{j}",
            transformer_config,
            num_tl,
        )

    if tensor_names.downsample_conv:
      stable_diffusion_loader._map_to_converted_state(
          state, tensor_names.downsample_conv,
          converted_state, f"down_encoders.{block_idx}.downsampler",
      )

  def _map_sdxl_mid_block(self, state, converted_state):
    residual_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM, group_num=32,
    )
    residual_activation_config = layers_cfg.ActivationConfig(
        layers_cfg.ActivationType.SILU
    )
    transformer_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.LAYER_NORM,
    )
    mid_ch = _BLOCK_OUT_CHANNELS[-1]

    # First resnet
    self._map_residual_block(
        state, converted_state,
        _mid_block_tensor_names.residual_block_tensor_names[0],
        "mid_block.resnets.0",
        unet_cfg.ResidualBlock2DConfig(
            in_channels=mid_ch, hidden_channels=mid_ch, out_channels=mid_ch,
            time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
            normalization_config=residual_norm_config,
            activation_config=residual_activation_config,
        ),
    )

    # Transformer
    num_heads = mid_ch // _HEAD_DIM
    transformer_config = _build_transformer_block_config(
        mid_ch, num_heads, 1,
        transformer_norm_config,
        layers_cfg.NormalizationConfig(
            layers_cfg.NormalizationType.GROUP_NORM,
            epsilon=1e-6, group_num=32,
        ),
        False,
    )
    self._map_multi_layer_transformer_block(
        state, converted_state,
        _mid_block_tensor_names.transformer_block_tensor_names[0],
        "mid_block.transformers.0",
        transformer_config,
        _MID_BLOCK_TRANSFORMER_LAYERS,
    )

    # Second resnet
    self._map_residual_block(
        state, converted_state,
        _mid_block_tensor_names.residual_block_tensor_names[1],
        "mid_block.resnets.1",
        unet_cfg.ResidualBlock2DConfig(
            in_channels=mid_ch, hidden_channels=mid_ch, out_channels=mid_ch,
            time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
            normalization_config=residual_norm_config,
            activation_config=residual_activation_config,
        ),
    )

  def _map_sdxl_up_decoder_block(
      self,
      state,
      converted_state,
      block_idx,
      tensor_names,
  ):
    reversed_channels = list(reversed(_BLOCK_OUT_CHANNELS))
    num_tl = _reversed_transformer_layers[block_idx]
    has_transformer = num_tl > 0
    out_channel = reversed_channels[block_idx]

    residual_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.GROUP_NORM, group_num=32,
    )
    residual_activation_config = layers_cfg.ActivationConfig(
        layers_cfg.ActivationType.SILU
    )
    transformer_norm_config = layers_cfg.NormalizationConfig(
        layers_cfg.NormalizationType.LAYER_NORM,
    )

    up_decoder_layers = _LAYERS_PER_BLOCK + 1
    for j in range(up_decoder_layers):
      self._map_residual_block(
          state, converted_state,
          tensor_names.residual_block_tensor_names[j],
          f"up_decoders.{block_idx}.resnets.{j}",
          unet_cfg.ResidualBlock2DConfig(
              in_channels=out_channel * 2,  # approximate, handled by skip
              hidden_channels=out_channel,
              out_channels=out_channel,
              time_embedding_channels=_TIME_EMBEDDING_BLOCKS_DIM,
              normalization_config=residual_norm_config,
              activation_config=residual_activation_config,
          ),
      )
      if has_transformer:
        num_heads = out_channel // _HEAD_DIM
        transformer_config = _build_transformer_block_config(
            out_channel, num_heads, 1,
            transformer_norm_config,
            layers_cfg.NormalizationConfig(
                layers_cfg.NormalizationType.GROUP_NORM,
                epsilon=1e-6, group_num=32,
            ),
            False,
        )
        self._map_multi_layer_transformer_block(
            state, converted_state,
            tensor_names.transformer_block_tensor_names[j],
            f"up_decoders.{block_idx}.transformers.{j}",
            transformer_config,
            num_tl,
        )

    if tensor_names.upsample_conv:
      stable_diffusion_loader._map_to_converted_state(
          state, tensor_names.upsample_conv,
          converted_state, f"up_decoders.{block_idx}.upsample_conv",
      )
