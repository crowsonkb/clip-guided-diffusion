# CLIP Guided Diffusion, version 2

A WIP updated version of my [2021 CLIP Guided Diffusion text to image method](https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA). Many options the original version supported are not added yet.

This new codebase uses a [k-diffusion](https://github.com/crowsonkb/k-diffusion) wrapper around [OpenAI's ImageNet diffusion models](https://github.com/crowsonkb/guided-diffusion) and uses a specialized solver, a splitting method adaptive step size version of [DPM-Solver++ SDE](https://github.com/LuChengTHU/dpm-solver).

## Installation

```
[clone this repository]
pip install -e [cloned repository]
```

## Usage

```clip_guided_diffusion [prompt] [options]```

Try:

```
clip_guided_diffusion "A beautiful fantasy painting of the wind by Wojciech Siudmak" --seed 15554
```

### Info tool

CLIP Guided Diffusion saves the prompt and CLI arguments used to generate an image in the image EXIF metadata. You can view this metadata with the `clip_guided_diffusion_info` tool.

### CLI arguments

- `--checkpoint`: the diffusion model checkpoint to load (default: None)

The diffusion model this repo uses if none is specified is https://models.rivershavewings.workers.dev/512x512_diffusion_uncond_finetune_008100.safetensors, which is the 512x512 OpenAI ImageNet diffusion model fine-tuned for 8100 steps to remove the class conditioning. Its SHA-256 hash is `02e212cbec7c9012eb12cd63fef6fa97640b4e8fcd6c6e1f410a52eea1925fe1`. It will auto-download it on first run.

- `--clip-model`: the CLIP model to use (default: ViT-B/16)

- `--clip-scale`: the CLIP guidance scale (default: 2500.0)

The CLIP guidance scale is automatically adjusted based on the image size, if you are generating a 1024x768 image it will be multiplied by (1024 * 768) / (512 * 512), or 3.

Raising the CLIP guidance scale will cause the solver to automatically decrease its step size and thus to do more steps.

- `--compile`: torch.compile() the model (default: False)

- `--cutn`: the number of random crops to use per step (default: 32)

- `--device`: the device to use (default: None)

It will autodetect CUDA or CPU if not specified.

- `--eta`: the multiplier for the noise variance. 0 gives ODE sampling, 1 gives standard diffusion SDE sampling. (default: 1.0)

`eta` higher than 1 is supported and produces a "soft" effect when raised to 1.5 or 2.

- `--init`: the initial image (default: None)

- `--init-sigma`: the starting noise level when using an init image (default: 10.0)

- `--max-cond`: the maximum amount that guidance is allowed to perturb a step (default: 0.05)

This is one of the two main options that controls the number of steps, and thus image quality. If the classifier guidance norm is larger than this on a step, the step size (h) will be adjusted downward from the maximum step size. Since classifier guidance perturbs steps more at the high noise levels, this option mainly applies "braking" to increase quality at the start of the noise schedule. Setting it lower increases quality, setting it higher decreases quality.

- `--max-h`: the maximum step size (default: 0.1)

This option sets the maximum step size. It mostly controls the step size at the end of the noise schedule. The units of step size are `log(sigma / sigma_next)`, where `sigma` is the noise standard deviation at the start of the step. Setting it lower increases quality, setting it higher decreases quality.

The total amount of time simulated for the OpenAI diffusion models is 9.66387, so setting max_h to 0.1 will do no fewer than 97 steps. At any point in the noise schedule, it is not allowed to take steps larger than a 97 step exponential schedule would.

- `--model-type`: the diffusion model type (eps or v) (default: eps)

- `--output`, `-o`: the output file (default: out.png)

- `--save-all`: save all intermediate denoised images (default: False)

- `--seed`: the random seed (default: 0)

- `--size`: the output size (default: (512, 512))

Image size is specified as (width, height).

- `--solver`: the SDE solver type (default: dpm3)

The supported SDE solver types are `euler`, `midpoint`, `heun`, and `dpm3`.

`euler` is first order and is also known as Euler Ancestral. It is the classic sampling method used for diffusion models. `midpoint` and `heun` are DPM-Solver++(2M) SDE subtypes. `dpm3` is a third order variant of DPM-Solver++ multistep.

The higher order solvers are [linear multistep methods](https://en.wikipedia.org/wiki/Linear_multistep_method): they reuse the model outputs of previous steps, one in the case of `midpoint` and `heun` and two in the case of `dpm3`. The higher order solvers are also splitting methods: they do not classifier guide the higher order correction, only the base first order step.

## Alternate models

You can also use [Emily S](https://twitter.com/nshepperd1)'s OpenImages fine-tune of the 512x512 ImageNet diffusion model, which is available at https://models.rivershavewings.workers.dev/512x512_diffusion_uncond_openimages_epoch28_withfilter.safetensors. It is a v objective model so you will need to specify `--model-type v`.

## To do

- Multiple prompts

- Image prompts

- LPIPS loss for init images

- Support the 256x256 ImageNet diffusion model

- CLIP model ensembling

- OpenCLIP support
