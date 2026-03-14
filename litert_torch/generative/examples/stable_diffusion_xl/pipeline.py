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

"""SDXL TFLite inference pipeline.

Supports dual text encoding (CLIP-L + OpenCLIP-G), additional conditioning
via add_embedding, and 1024x1024 default resolution.
"""

import argparse
import os
import pathlib
from typing import Optional

import litert_torch
from litert_torch.generative.examples.stable_diffusion import samplers
from litert_torch.generative.examples.stable_diffusion import tokenizer
from litert_torch.generative.examples.stable_diffusion_xl import util
import numpy as np
from PIL import Image
import torch
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--tokenizer_vocab_dir', type=str, required=True,
    help='Directory to the tokenizer vocabulary files (merges.txt, vocab.json)',
)
arg_parser.add_argument(
    '--clip_ckpt', type=str, required=True,
    help='Path to CLIP-L TFLite file',
)
arg_parser.add_argument(
    '--open_clip_ckpt', type=str, required=True,
    help='Path to OpenCLIP-G TFLite file',
)
arg_parser.add_argument(
    '--diffusion_ckpt', type=str, required=True,
    help='Path to SDXL diffusion TFLite file',
)
arg_parser.add_argument(
    '--decoder_ckpt', type=str, required=True,
    help='Path to SDXL decoder TFLite file',
)
arg_parser.add_argument(
    '--output_path', type=str, required=True,
    help='Path to the output generated image file.',
)
arg_parser.add_argument(
    '--prompt', type=str,
    default='a photograph of an astronaut riding a horse',
    help='The prompt to guide the image generation.',
)
arg_parser.add_argument(
    '--n_inference_steps', type=int, default=20,
    help='The number of denoising steps.',
)
arg_parser.add_argument(
    '--sampler', type=str, default='k_euler',
    choices=['k_euler', 'k_euler_ancestral', 'k_lms'],
    help='Sampler for denoising.',
)
arg_parser.add_argument(
    '--seed', type=int, default=None,
    help='Random seed for deterministic generation.',
)


class StableDiffusionXL:
  """SDXL TFLite model pipeline with dual text encoders."""

  def __init__(
      self,
      *,
      tokenizer_vocab_dir: str,
      clip_ckpt: str,
      open_clip_ckpt: str,
      diffusion_ckpt: str,
      decoder_ckpt: str,
      text_projection: Optional[np.ndarray] = None,
  ):
    self.tokenizer = tokenizer.Tokenizer(tokenizer_vocab_dir)
    self.clip = litert_torch.model.TfLiteModel.load(clip_ckpt)
    self.open_clip = litert_torch.model.TfLiteModel.load(open_clip_ckpt)
    self.diffusion = litert_torch.model.TfLiteModel.load(diffusion_ckpt)
    self.decoder = litert_torch.model.TfLiteModel.load(decoder_ckpt)
    self.text_projection = text_projection


def run_sdxl_tflite_pipeline(
    model: StableDiffusionXL,
    prompt: str,
    output_path: str,
    uncond_prompt: Optional[str] = None,
    cfg_scale: float = 7.5,
    height: int = 1024,
    width: int = 1024,
    sampler: str = 'k_euler',
    n_inference_steps: int = 20,
    seed: Optional[int] = None,
):
  """Run SDXL pipeline with TFLite models.

  Args:
    model: StableDiffusionXL model with dual text encoders.
    prompt: Text prompt for image generation.
    output_path: Output image path.
    uncond_prompt: Negative prompt (default: empty).
    cfg_scale: Classifier-free guidance scale.
    height: Image height (default: 1024).
    width: Image width (default: 1024).
    sampler: Sampler name.
    n_inference_steps: Number of denoising steps.
    seed: Random seed.
  """
  if height % 8 or width % 8:
    raise ValueError('height and width must be a multiple of 8')
  if seed is not None:
    np.random.seed(seed)
  if uncond_prompt is None:
    uncond_prompt = ''

  if sampler == 'k_lms':
    sampler = samplers.KLMSSampler(n_inference_steps=n_inference_steps)
  elif sampler == 'k_euler':
    sampler = samplers.KEulerSampler(n_inference_steps=n_inference_steps)
  elif sampler == 'k_euler_ancestral':
    sampler = samplers.KEulerAncestralSampler(
        n_inference_steps=n_inference_steps
    )
  else:
    raise ValueError(f'Unknown sampler: {sampler}')

  # Dual text encoding
  cond_tokens = np.array(model.tokenizer.encode(prompt)).astype(np.int32)
  uncond_tokens = np.array(model.tokenizer.encode(uncond_prompt)).astype(np.int32)

  # CLIP-L encoding (hidden states before final norm)
  cond_clip = model.clip(cond_tokens, signature_name='encode')
  uncond_clip = model.clip(uncond_tokens, signature_name='encode')

  # OpenCLIP-G encoding (penultimate hidden states + final normed states)
  cond_open_clip_hidden, cond_final = model.open_clip(
      cond_tokens, signature_name='encode'
  )
  uncond_open_clip_hidden, uncond_final = model.open_clip(
      uncond_tokens, signature_name='encode'
  )

  # EOS pooling + text_projection (done outside TFLite model for compatibility)
  cond_eos_idx = int(np.argmax(cond_tokens))
  uncond_eos_idx = int(np.argmax(uncond_tokens))
  cond_pooled = cond_final[0, cond_eos_idx] @ model.text_projection
  uncond_pooled = uncond_final[0, uncond_eos_idx] @ model.text_projection
  cond_pooled = cond_pooled[np.newaxis, :]
  uncond_pooled = uncond_pooled[np.newaxis, :]

  # Concatenate CLIP-L (768) + OpenCLIP-G (1280) = 2048 context
  cond_context = np.concatenate([cond_clip, cond_open_clip_hidden], axis=-1)
  uncond_context = np.concatenate([uncond_clip, uncond_open_clip_hidden], axis=-1)
  context = np.concatenate([cond_context, uncond_context], axis=0)

  # Add time ids embedding
  add_time_ids = util.get_add_time_ids(
      original_size=(height, width),
      crop_coords=(0, 0),
      target_size=(height, width),
  )
  time_ids_emb = util.encode_add_time_ids(add_time_ids)

  # Additional embedding: pooled_text_embed + time_ids_embed
  cond_add_emb = np.concatenate(
      [cond_pooled, time_ids_emb.numpy()], axis=-1
  ).astype(np.float32)
  uncond_add_emb = np.concatenate(
      [uncond_pooled, time_ids_emb.numpy()], axis=-1
  ).astype(np.float32)
  add_emb = np.concatenate([cond_add_emb, uncond_add_emb], axis=0)

  # Initialize latents
  noise_shape = (1, 4, height // 8, width // 8)
  latents = np.random.normal(size=noise_shape).astype(np.float32)
  latents *= sampler.initial_scale

  # Diffusion loop
  timesteps = tqdm.tqdm(sampler.timesteps)
  for _, timestep in enumerate(timesteps):
    time_embedding = util.get_time_embedding(timestep)

    input_latents = latents * sampler.get_input_scale()
    input_latents = input_latents.repeat(2, axis=0)

    output = model.diffusion(
        input_latents.astype(np.float32),
        context.astype(np.float32),
        time_embedding.numpy().astype(np.float32),
        add_emb.astype(np.float32),
        signature_name='diffusion',
    )

    output_cond, output_uncond = np.split(output, 2, axis=0)
    output = cfg_scale * (output_cond - output_uncond) + output_uncond
    latents = sampler.step(latents, output)

  # Image decoding
  images = model.decoder(latents.astype(np.float32), signature_name='decode')
  images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
  images = util.move_channel(images, to='last')
  if not os.path.exists(output_path):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(images[0].astype(np.uint8)).save(output_path)


if __name__ == '__main__':
  args = arg_parser.parse_args()
  run_sdxl_tflite_pipeline(
      StableDiffusionXL(
          tokenizer_vocab_dir=args.tokenizer_vocab_dir,
          clip_ckpt=args.clip_ckpt,
          open_clip_ckpt=args.open_clip_ckpt,
          diffusion_ckpt=args.diffusion_ckpt,
          decoder_ckpt=args.decoder_ckpt,
      ),
      prompt=args.prompt,
      output_path=args.output_path,
      sampler=args.sampler,
      n_inference_steps=args.n_inference_steps,
      seed=args.seed,
  )
