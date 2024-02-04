# CLIP Guided Diffusion, version 2

A WIP updated version of my [2021 CLIP Guided Diffusion text to image method](https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA). Many options the original version supported are not added yet.

This new codebase uses a [k-diffusion](https://github.com/crowsonkb/k-diffusion) wrapper around [OpenAI's ImageNet diffusion models](https://github.com/crowsonkb/guided-diffusion) and uses a specialized solver, a splitting method adaptive step size version of [DPM-Solver++ SDE](https://github.com/LuChengTHU/dpm-solver).

## Example outputs

<img src="https://raw.githubusercontent.com/crowsonkb/clip-guided-diffusion/master/assets/example_1.png" width=640 height=512></img>

*Prompt: chrysanthemum supernova, trending on ArtStation*

<img src="https://raw.githubusercontent.com/crowsonkb/clip-guided-diffusion/master/assets/example_2.png" width=640 height=512></img>

*Prompt: A landscape resembling the Death tarot card by Gerardo Dottori*

## Installation

### Clone repository

```bash
git clone https://github.com/Birch-san/clip-guided-diffusion.git
cd clip-guided-diffusion
```

### Create & activate a virtual environment

```bash
python3 -m venv venv
. venv/bin/activate
# setup will fail to resolve modules such as torch unless you have wheel package
pip install wheel
```

### Install NATTEN

With your clip-guided-diffusion virtualenv active, do one of the following:

**[Option 1] Try installing wheel:**

You could _try_ installing a wheel of NATTEN, but for me this still attempted a build-from-source (and failed due to not finding CUDA compiler).

```bash
pip install natten -f https://shi-labs.com/natten/wheels/cu122/torch2.2.0/index.html
```

**[Option 2] Build from source:**

Clone NATTEN somewhere (ideally not inside this repository):

```bash
git clone https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
```

Install cmake for your OS if you don't already have it, and install in your venv the version of cmake that NATTEN prefers. I tried newer cmake myself and it seemed to have trouble locating CUDA, so that might be why they pinned an older version.

```bash
sudo apt-get install cmake
pip install cmake==3.20.3
```

Build NATTEN from source

```bash
CUDACXX=/usr/local/cuda/bin/nvcc make install CUDA_ARCH="8.9" WORKERS=4
```

### Install the rest of the dependencies

Go back to the clip-guided-diffusion repository, ensure your virtualenv is activate, and install the remaining dependencies:

```bash
pip install .
```

## Usage

```bash
clip_guided_diffusion [prompt] [options]
```

Try:

```bash
clip_guided_diffusion "A beautiful fantasy painting of the wind by Wojciech Siudmak" --seed 15554
```

### Multiple prompts

Multiple prompts can be specified by separating them with `|`, and relative weights per prompt can be specified by appending `:` and the weight to the prompt. For example:

```bash
clip_guided_diffusion "First prompt:1|Second prompt:2"
```

### Info tool

CLIP Guided Diffusion saves the prompt and CLI arguments used to generate an image in the image EXIF metadata. You can view this metadata with the `clip_guided_diffusion_info` tool.

### CLI arguments

- `--checkpoint`: the diffusion model checkpoint to load (default: None)

The diffusion model this repo uses if none is specified is https://models.rivershavewings.workers.dev/512x512_diffusion_uncond_finetune_008100.safetensors, which is the 512x512 OpenAI ImageNet diffusion model fine-tuned for 8100 steps to remove the class conditioning. Its SHA-256 hash is `02e212cbec7c9012eb12cd63fef6fa97640b4e8fcd6c6e1f410a52eea1925fe1`. It will auto-download it on first run.

- `--clip-model`: the CLIP model(s) to use (default: ViT-B/16)

- `--clip-scale`: the CLIP guidance scale(s) (default: 2000.0)

The CLIP guidance scale is automatically adjusted based on the image size, if you are generating a 1024x768 image it will be multiplied by (1024 * 768) / (512 * 512), or 3.

Raising the CLIP guidance scale will cause the solver to automatically decrease its step size and thus to do more steps.

- `--compile`: torch.compile() the model (default: False)

- `--cutn`: the number of random crops (per CLIP model) to use per step (default: 32)

- `--device`: the device to use (default: None)

It will autodetect CUDA or CPU if not specified.

- `--eta`: the multiplier for the noise variance. 0 gives ODE sampling, 1 gives standard diffusion SDE sampling. (default: 1.0)

`eta` higher than 1 is supported and produces a "soft" effect when raised to 1.5 or 2.

- `--image-prompts`: the image prompts and weights (path to image and weight separated by colons) to use (default: None)

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

- LPIPS loss for init images

- Support the 256x256 ImageNet diffusion model

- OpenCLIP support
