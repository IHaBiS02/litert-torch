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

"""CLIP-L text encoder for SDXL.

Reuses the same CLIP architecture as SD 1.5 but with SDXL checkpoint tensor names
(conditioner.embedders.0.*) and returns hidden states before final layer norm
(required for SDXL's dual encoder concatenation).
"""

from litert_torch.generative.layers.attention import TransformerBlock
import litert_torch.generative.layers.attention_utils as attention_utils
import litert_torch.generative.layers.builder as builder
import litert_torch.generative.layers.model_config as cfg
import litert_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj=(
        "conditioner.embedders.0.transformer.text_model.encoder.layers.{}.mlp.fc1"
    ),
    ff_down_proj=(
        "conditioner.embedders.0.transformer.text_model.encoder.layers.{}.mlp.fc2"
    ),
    attn_query_proj="conditioner.embedders.0.transformer.text_model.encoder.layers.{}.self_attn.q_proj",
    attn_key_proj="conditioner.embedders.0.transformer.text_model.encoder.layers.{}.self_attn.k_proj",
    attn_value_proj="conditioner.embedders.0.transformer.text_model.encoder.layers.{}.self_attn.v_proj",
    attn_output_proj="conditioner.embedders.0.transformer.text_model.encoder.layers.{}.self_attn.out_proj",
    pre_attn_norm=(
        "conditioner.embedders.0.transformer.text_model.encoder.layers.{}.layer_norm1"
    ),
    post_attn_norm=(
        "conditioner.embedders.0.transformer.text_model.encoder.layers.{}.layer_norm2"
    ),
    embedding=(
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding"
    ),
    embedding_position="conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
    final_norm="conditioner.embedders.0.transformer.text_model.final_layer_norm",
    lm_head=None,
)


class CLIPEncoder(nn.Module):
  """CLIP-L text encoder for SDXL.

  Returns hidden states before final layer norm, which is concatenated
  with OpenCLIP-G output to form the 2048-dim context for the SDXL UNet.
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
    self.mask_cache = attention_utils.build_causal_mask_cache(
        size=config.max_seq_len, dtype=torch.float32
    )

  @torch.inference_mode
  def forward(self, tokens: torch.IntTensor) -> torch.FloatTensor:
    """Forward pass returning hidden states before final norm.

    Args:
        tokens: Input token ids of shape (batch_size, seq_len).

    Returns:
        Hidden states of shape (batch_size, seq_len, 768) before final norm.
    """
    state = self.tok_embedding(tokens) + self.tok_embedding_position
    for layer in self.transformer_blocks:
      state = layer(state, mask=self.mask_cache)
    # SDXL uses hidden states BEFORE final layer norm for CLIP-L
    return state


def get_model_config() -> cfg.ModelConfig:
  """Get configs for CLIP-L in SDXL (same architecture as SD 1.5)."""
  max_seq_len = 77
  vocab_size = 49408
  num_layers = 12
  num_heads = 12
  num_query_groups = 12
  embedding_dim = 768

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
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_QUICK),
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
