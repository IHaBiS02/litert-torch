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

"""Convert SDXL models to TFLite format.

Converts 4 components: CLIP-L, OpenCLIP-G, SDXL UNet, VAE Decoder.
"""

import os
import pathlib

from absl import app
from absl import flags
import litert_torch
from litert_torch.generative.examples.stable_diffusion_xl import clip
from litert_torch.generative.examples.stable_diffusion_xl import decoder
from litert_torch.generative.examples.stable_diffusion_xl import diffusion
from litert_torch.generative.examples.stable_diffusion_xl import open_clip
from litert_torch.generative.examples.stable_diffusion_xl import util
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.utilities import stable_diffusion_loader
import torch

_CKPT = flags.DEFINE_string(
    'ckpt',
    os.path.join(
        pathlib.Path.home(),
        'Downloads/stable-diffusion-xl/sd_xl_base_1.0.safetensors',
    ),
    help='Path to SDXL base checkpoint (safetensors)',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/sdxl_tflite',
    help='Path to the converted TFLite directory.',
)

_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    help='Whether to quantize the model during conversion.',
    default=False,
)

_DEVICE_TYPE = flags.DEFINE_string(
    'device_type',
    'gpu',
    help='The device type of the model. Currently supported: cpu, gpu.',
)


@torch.inference_mode
def convert_sdxl_to_tflite(
    output_dir: str,
    ckpt_path: str,
    image_height: int = 1024,
    image_width: int = 1024,
    quantize: bool = True,
    device_type: str = "cpu",
):
  """Convert all SDXL components to TFLite.

  Args:
    output_dir: Output directory for TFLite files.
    ckpt_path: Path to SDXL safetensors checkpoint.
    image_height: Target image height.
    image_width: Target image width.
    quantize: Whether to apply weight-only INT8 quantization.
    device_type: Device type (cpu/gpu).
  """
  if not os.path.exists(output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

  quant_config = quant_recipes.full_weight_only_recipe() if quantize else None

  # --- CLIP-L ---
  print("Converting CLIP-L text encoder...")
  clip_model = clip.CLIPEncoder(clip.get_model_config())
  clip_loader = stable_diffusion_loader.ClipModelLoader(
      ckpt_path, clip.TENSOR_NAMES
  )
  clip_loader.load(clip_model, strict=False)
  clip_model.eval()

  n_tokens = 77
  prompt_tokens = torch.full((1, n_tokens), 0, dtype=torch.int)

  litert_torch.signature('encode', clip_model, (prompt_tokens,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/clip.tflite')
  print(f"  Saved: {output_dir}/clip.tflite")

  # --- OpenCLIP-G ---
  print("Converting OpenCLIP-G text encoder...")
  open_clip_model = open_clip.OpenCLIP(open_clip.get_model_config())
  open_clip_loader = open_clip.OpenCLIPModelLoader(ckpt_path)
  open_clip_loader.load(open_clip_model, strict=False)
  open_clip_model.eval()

  litert_torch.signature('encode', open_clip_model, (prompt_tokens,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/open_clip.tflite')
  print(f"  Saved: {output_dir}/open_clip.tflite")

  # --- SDXL UNet ---
  print("Converting SDXL UNet...")
  diffusion_model = diffusion.SDXLDiffusion(
      batch_size=2, device_type=device_type
  )
  diffusion_loader = diffusion.SDXLDiffusionModelLoader(ckpt_path)
  diffusion_loader.load(diffusion_model, strict=False)
  diffusion_model.eval()

  input_latents = torch.zeros(
      (1, 4, image_height // 8, image_width // 8), dtype=torch.float32
  )

  # Get context shape from CLIP-L + OpenCLIP-G
  clip_out = clip_model(prompt_tokens)
  open_clip_out, pooled_out = open_clip_model(prompt_tokens)
  context_cond = torch.cat([clip_out, open_clip_out], dim=-1)
  context_uncond = torch.zeros_like(context_cond)
  context = torch.cat([context_cond, context_uncond], axis=0)

  time_embedding = util.get_time_embedding(0)

  # Add embedding
  add_time_ids = util.get_add_time_ids(
      original_size=(image_height, image_width),
      crop_coords=(0, 0),
      target_size=(image_height, image_width),
  )
  time_ids_emb = util.encode_add_time_ids(add_time_ids)
  add_emb_cond = torch.cat([pooled_out, time_ids_emb], dim=-1)
  add_emb_uncond = torch.zeros_like(add_emb_cond)
  add_emb = torch.cat([add_emb_cond, add_emb_uncond], axis=0)

  litert_torch.signature(
      'diffusion',
      diffusion_model,
      (
          torch.repeat_interleave(input_latents, 2, 0),
          context,
          time_embedding,
          add_emb,
      ),
  ).convert(quant_config=quant_config).export(
      f'{output_dir}/diffusion.tflite'
  )
  print(f"  Saved: {output_dir}/diffusion.tflite")

  # --- VAE Decoder ---
  print("Converting VAE Decoder...")
  decoder_model = decoder.Decoder(
      decoder.get_model_config(device_type=device_type)
  )
  decoder_loader = stable_diffusion_loader.AutoEncoderModelLoader(
      ckpt_path, decoder.TENSOR_NAMES
  )
  decoder_loader.load(decoder_model, strict=False)
  decoder_model.eval()

  litert_torch.signature('decode', decoder_model, (input_latents,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/decoder.tflite')
  print(f"  Saved: {output_dir}/decoder.tflite")

  print("\nAll SDXL components converted successfully!")


def main(_):
  convert_sdxl_to_tflite(
      output_dir=_OUTPUT_DIR.value,
      ckpt_path=_CKPT.value,
      quantize=_QUANTIZE.value,
      device_type=_DEVICE_TYPE.value,
  )


if __name__ == '__main__':
  app.run(main)
