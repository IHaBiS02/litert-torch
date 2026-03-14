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

"""OpenCLIP-G (ViT-bigG) text encoder for SDXL.

32 layers, 1280 embedding dim, 20 attention heads, GELU activation.
Returns (penultimate_hidden_states, pooled_output) where:
  - penultimate_hidden_states: output from second-to-last transformer layer
  - pooled_output: EOS token representation projected via text_projection

Checkpoint keys use fused QKV format (in_proj_weight/in_proj_bias) and
a custom prefix (conditioner.embedders.1.model.*).
"""

from typing import Dict, List, Tuple

from litert_torch.generative.layers.attention import TransformerBlock
import litert_torch.generative.layers.attention_utils as attention_utils
import litert_torch.generative.layers.builder as builder
import litert_torch.generative.layers.model_config as cfg
import litert_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


class OpenCLIP(nn.Module):
  """OpenCLIP-G text encoder for SDXL.

  Architecture: 32-layer transformer, 1280-dim, 20 heads.
  Returns penultimate hidden states and pooled output.
  """

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    self.tok_embedding_position = nn.Parameter(
        torch.zeros((config.max_seq_len, config.embedding_dim)),
        requires_grad=False,
    )
    self.config = config
    block_config = config.block_config(0)
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(block_config, config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )
    self.text_projection = nn.Parameter(
        torch.zeros((config.embedding_dim, config.embedding_dim)),
        requires_grad=False,
    )
    self.mask_cache = attention_utils.build_causal_mask_cache(
        size=config.max_seq_len, dtype=torch.float32
    )

  @torch.inference_mode
  def forward(
      self, tokens: torch.IntTensor
  ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Forward pass returning penultimate hidden states and final normed states.

    EOS pooling and text_projection are applied outside the model (in the
    pipeline) because argmax and fancy indexing are not TFLite-compatible.

    Args:
        tokens: Input token ids of shape (batch_size, seq_len).

    Returns:
        Tuple of:
          - penultimate_hidden_states: (batch_size, seq_len, 1280)
          - final_hidden_states: (batch_size, seq_len, 1280) after final norm
    """
    state = self.tok_embedding(tokens) + self.tok_embedding_position
    penultimate = None
    for i, layer in enumerate(self.transformer_blocks):
      state = layer(state, mask=self.mask_cache)
      if i == self.config.num_layers - 2:
        penultimate = state
    state = self.final_norm(state)

    return penultimate, state


def get_model_config() -> cfg.ModelConfig:
  """Get configs for OpenCLIP-G in SDXL."""
  max_seq_len = 77
  vocab_size = 49408
  num_layers = 32
  num_heads = 20
  num_query_groups = 20
  embedding_dim = 1280

  attn_config = cfg.AttentionConfig(
      num_heads=num_heads,
      head_dim=embedding_dim // num_heads,
      num_query_groups=num_query_groups,
      rotary_base=0,
      rotary_percentage=0.0,
      qkv_use_bias=True,
      qkv_transpose_before_split=True,
      qkv_fused_interleaved=False,
      output_proj_use_bias=True,
      enable_kv_cache=False,
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
      intermediate_size=embedding_dim * 4,
      use_bias=True,
  )

  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.LAYER_NORM, enable_hlfb=False
  )

  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )

  config = cfg.ModelConfig(
      vocab_size=vocab_size,
      num_layers=num_layers,
      max_seq_len=max_seq_len,
      embedding_dim=embedding_dim,
      block_configs=block_config,
      final_norm_config=norm_config,
  )

  return config


class OpenCLIPModelLoader(loading_utils.ModelLoader):
  """Custom loader for OpenCLIP-G that handles fused QKV (in_proj_weight) format."""

  def __init__(self, file_name: str):
    self._file_name = file_name
    self._loader = self._get_loader()

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    state = self._loader(self._file_name)
    converted_state = dict()
    prefix = "conditioner.embedders.1.model"

    # Embeddings
    converted_state["tok_embedding.weight"] = state.pop(
        f"{prefix}.token_embedding.weight"
    )
    converted_state["tok_embedding_position"] = state.pop(
        f"{prefix}.positional_embedding"
    )

    # Final norm
    converted_state["final_norm.weight"] = state.pop(
        f"{prefix}.ln_final.weight"
    )
    converted_state["final_norm.bias"] = state.pop(
        f"{prefix}.ln_final.bias"
    )

    # Text projection
    converted_state["text_projection"] = state.pop(
        f"{prefix}.text_projection"
    )

    # Transformer blocks
    for i in range(model.config.num_layers):
      block_prefix = f"{prefix}.transformer.resblocks.{i}"
      model_prefix = f"transformer_blocks.{i}"

      # Pre-attention norm (ln_1)
      converted_state[f"{model_prefix}.pre_atten_norm.weight"] = state.pop(
          f"{block_prefix}.ln_1.weight"
      )
      converted_state[f"{model_prefix}.pre_atten_norm.bias"] = state.pop(
          f"{block_prefix}.ln_1.bias"
      )

      # Post-attention norm (ln_2)
      converted_state[f"{model_prefix}.post_atten_norm.weight"] = state.pop(
          f"{block_prefix}.ln_2.weight"
      )
      converted_state[f"{model_prefix}.post_atten_norm.bias"] = state.pop(
          f"{block_prefix}.ln_2.bias"
      )

      # Fused QKV attention (in_proj_weight -> qkv_projection)
      converted_state[
          f"{model_prefix}.atten_func.qkv_projection.weight"
      ] = state.pop(f"{block_prefix}.attn.in_proj_weight")
      converted_state[
          f"{model_prefix}.atten_func.qkv_projection.bias"
      ] = state.pop(f"{block_prefix}.attn.in_proj_bias")

      # Output projection
      converted_state[
          f"{model_prefix}.atten_func.output_projection.weight"
      ] = state.pop(f"{block_prefix}.attn.out_proj.weight")
      converted_state[
          f"{model_prefix}.atten_func.output_projection.bias"
      ] = state.pop(f"{block_prefix}.attn.out_proj.bias")

      # Feed forward (MLP)
      converted_state[f"{model_prefix}.ff.w1.weight"] = state.pop(
          f"{block_prefix}.mlp.c_fc.weight"
      )
      converted_state[f"{model_prefix}.ff.w1.bias"] = state.pop(
          f"{block_prefix}.mlp.c_fc.bias"
      )
      converted_state[f"{model_prefix}.ff.w2.weight"] = state.pop(
          f"{block_prefix}.mlp.c_proj.weight"
      )
      converted_state[f"{model_prefix}.ff.w2.bias"] = state.pop(
          f"{block_prefix}.mlp.c_proj.bias"
      )

    if strict and state:
      raise ValueError(
          f"Failed to map all tensors. Remaining: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)
