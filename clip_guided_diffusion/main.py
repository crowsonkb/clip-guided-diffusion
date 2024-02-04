"""Generates images from text prompts with CLIP guided diffusion."""

import argparse
from concurrent import futures
from functools import partial
import json
import hashlib
import math
from pathlib import Path
from typing import Literal
import safetensors.torch as safetorch

import clip
import k_diffusion as K
from PIL import ExifTags, Image
import requests
from rich import print
from rich.align import Align
from rich.panel import Panel
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

repo_root = Path(__file__).parents[1]
config_dir = repo_root / 'config'

print = tqdm.external_write_mode()(print)
srgb_profile = (Path(__file__).resolve().parent / "sRGB Profile.icc").read_bytes()


def download_file(url, root, expected_sha256):
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(url).name

    if target.exists() and not target.is_file():
        raise RuntimeError(f"{target} exists and is not a regular file")

    if target.is_file():
        if hashlib.sha256(open(target, "rb").read()).hexdigest() == expected_sha256:
            return target
        else:
            print(
                f"{target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    response = requests.get(url, stream=True)
    with open(target, "wb") as output:
        size = int(response.headers.get("content-length", 0))
        with tqdm(total=size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=8192):
                output.write(data)
                pbar.update(len(data))

    if hashlib.sha256(open(target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return target


class JsonEncoderForMakerNote(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, argparse.Namespace):
            return vars(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return obj


def save_image(image, path, prompt=None, args=None, steps=None):
    if isinstance(image, torch.Tensor):
        image = K.utils.to_pil_image(image)
    exif = image.getexif()
    exif[ExifTags.Base.Software] = "CLIP Guided Diffusion"
    if prompt is not None:
        exif[ExifTags.Base.ImageDescription] = prompt
    obj = {}
    if args is not None:
        obj["args"] = args
    if steps is not None:
        obj["steps"] = steps
    exif[ExifTags.Base.MakerNote] = json.dumps(obj, cls=JsonEncoderForMakerNote)
    image.save(path, exif=exif, icc_profile=srgb_profile)


def projx(x):
    return x / x.norm(dim=-1, keepdim=True)


def proju(x, u):
    return u - torch.sum(x * u, dim=-1, keepdim=True) * x


def dist(u, v, keepdim=False):
    norm = torch.linalg.norm(u - v, dim=-1, keepdim=keepdim)
    return 2 * torch.arcsin(norm / 2)


def retr(x, u):
    return projx(x + u)


def logmap(x, y):
    u = proju(x, y - x)
    d = dist(x, y, keepdim=True)
    result = u * d / u.norm(dim=-1, keepdim=True)
    return torch.where(d > 1e-4, result, u)


class SphericalAverageError(Exception):
    pass


def spherical_average(p, w=None, tol=1e-4):
    if p.dtype in {torch.float16, torch.bfloat16}:
        p = p.float()
    if w is None:
        w = p.new_ones(p.shape[:-1])
    if p.shape[:-1] != w.shape:
        s1, s2, s3 = tuple(p.shape[:-1]), tuple(p.shape), tuple(w.shape)
        raise ValueError(f"expected w shape {s1} for p shape {s2}, got {s3}")
    w = w / w.sum(dim=-1, keepdim=True)
    w = w.unsqueeze(-1)
    p = projx(p)
    q = projx(torch.sum(p * w, dim=-2))
    norm_prev = p.new_tensor(float("inf"))
    while True:
        p_star = logmap(q.unsqueeze(-2), p)
        rgrad = torch.sum(p_star * w, dim=-2)
        q = retr(q, rgrad)
        norm = rgrad.detach().norm(dim=-1).max()
        if not norm.isfinite():
            raise SphericalAverageError("grad norm is not finite")
        if norm >= norm_prev:
            raise SphericalAverageError("grad norm did not decrease")
        if norm <= tol:
            break
        norm_prev = norm
    return q


def batch_crop(x, out_size, corners, mode="bilinear", padding_mode="zeros"):
    # batch crops out of a single image and resize them all to out_size
    # x, the input image, is NCHW but N must be 1
    # out_size is a tuple, (out_h, out_w)
    # crop corners tensor is N x <0 for h, 1 for w> x <0 for start loc, 1 for end loc>
    n, c, h, w = x.shape
    if n != 1:
        raise ValueError("batch_crop() only works with a single image")
    # make base grid, <0 for h, 1 for w> x H x W
    ramp_h = torch.linspace(0, 1, out_size[0], device=x.device)
    ramp_w = torch.linspace(0, 1, out_size[1], device=x.device)
    grid = torch.stack(torch.meshgrid(ramp_h, ramp_w, indexing="ij"), dim=-1)
    # scale corners tensor to the -1 to 1 range used by grid_sample()
    corners = corners / corners.new_tensor([h - 1, w - 1])[None, :, None] * 2 - 1
    # work out the values to scale and shift the h and w grids by
    scales = corners[:, :, 1] - corners[:, :, 0]
    shifts = corners[:, :, 0]
    # scale and shift the grids
    grid = grid[None] * scales[:, None, None, :] + shifts[:, None, None, :]
    # resize and crop
    x = x.expand([corners.shape[0], -1, -1, -1])
    grid = grid.flip(3)
    return F.grid_sample(x, grid, mode, padding_mode, align_corners=False)


def stratified_sample(strata_begin, strata_end, shuffle=True):
    assert strata_begin.shape == strata_end.shape
    assert strata_begin.ndim == 1
    assert strata_begin.device == strata_end.device
    assert strata_begin.dtype == strata_end.dtype

    n_strata = strata_begin.shape[0]
    device = strata_begin.device
    dtype = strata_begin.dtype

    u = torch.rand([n_strata], dtype=dtype, device=device)
    samples = u * (strata_end - strata_begin) + strata_begin
    if shuffle:
        samples = samples[torch.randperm(n_strata, device=device)]

    return samples


def mean_pad(x, pad):
    x_zero_pad = F.pad(x, pad, "constant")
    mask = F.pad(torch.zeros_like(x), pad, "constant", 1.0)
    return x_zero_pad + mask * x.mean(dim=[2, 3], keepdim=True)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class CLIPWrapper(nn.Module):
    def __init__(self, model, preprocess, cutn=32):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.cutn = cutn

    @property
    def cut_size(self):
        return (self.model.visual.input_resolution,) * 2

    @classmethod
    def from_pretrained(cls, clip_name, device="cpu", jit=False, **kwargs):
        model = clip.load(clip_name, device=device, jit=jit)[0].eval().requires_grad_(False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        preprocess = Normalize(mean * 2 - 1, std * 2).to(device)
        return cls(model, preprocess, **kwargs)

    def encode_image(self, x, jitter=16):
        if x.ndim == 3:
            x = x[None]
        n, c, h, w = x.shape
        x = self.preprocess(x)

        # Resize image
        aspect = w / h
        if w > h:
            new_h = self.cut_size[0] + jitter
            new_w = round(new_h * aspect)
        else:
            new_w = self.cut_size[1] + jitter
            new_h = round(new_w / aspect)
        x = F.interpolate(
            x, (new_h, new_w), mode="bicubic", align_corners=False, antialias=True
        )

        # Crop image
        jitterx = torch.rand([self.cutn], device=x.device) * jitter
        jittery = torch.rand([self.cutn], device=x.device) * jitter
        offsetx = torch.linspace(
            0, new_w - self.cut_size[1], self.cutn, device=x.device
        )
        offsety = torch.linspace(
            0, new_h - self.cut_size[0], self.cutn, device=x.device
        )
        offsets = torch.stack([offsety + jittery, offsetx + jitterx], dim=-1)
        corners = torch.stack([offsets, offsets + x.new_tensor(self.cut_size)], dim=-1)
        x = batch_crop(x, self.cut_size, corners)

        image_embeds = self.model.encode_image(x)
        return spherical_average(image_embeds)

    def encode_text(self, s):
        toks = clip.tokenize(s, truncate=True).to(self.model.logit_scale.device)
        return self.model.encode_text(toks).float()

    def forward(self, x):
        n, c, h, w = x.shape
        min_size = min(self.cut_size)
        max_size = min(w, h)
        pad_size = max(w, h)
        pad_w, pad_h = (pad_size - w) // 2, (pad_size - h) // 2
        x = mean_pad(x, (pad_w, pad_w, pad_h, pad_h))

        # Stratified sampling of crop sizes
        dist = torch.distributions.Normal(0.8 * max_size, 0.3 * max_size)
        strata = torch.linspace(
            dist.cdf(x.new_tensor(min_size)),
            dist.cdf(x.new_tensor(pad_size)),
            self.cutn + 1,
            device=x.device,
        )
        size = dist.icdf(stratified_sample(strata[:-1], strata[1:]))

        # Uniform sampling of crop offsets
        offsetx = torch.rand([self.cutn], device=x.device) * (pad_size - size)
        offsety = torch.rand([self.cutn], device=x.device) * (pad_size - size)
        offsets = torch.stack([offsety, offsetx], dim=-1)
        corners = torch.stack([offsets, offsets + size[:, None]], dim=-1)

        x = self.preprocess(x)
        cutouts = batch_crop(x, self.cut_size, corners)
        image_embeds = self.model.encode_image(cutouts)
        return spherical_average(image_embeds)


@torch.no_grad()
def sample_dpm_guided(
    model,
    x,
    sigma_min,
    sigma_max,
    max_h,
    max_cond,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    callback=None,
):
    """DPM-Solver++(1/2/3M) SDE (Kat's splitting version)."""
    noise_sampler = (
        K.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
        if noise_sampler is None
        else noise_sampler
    )
    if solver_type not in {"euler", "midpoint", "heun", "dpm3"}:
        raise ValueError('solver_type must be "euler", "midpoint", "heun", or "dpm3"')

    # Helper functions
    def sigma_to_t(sigma):
        return -torch.log(sigma)

    def t_to_sigma(t):
        return torch.exp(-t)

    def phi_1(h):
        return torch.expm1(-h)

    def h_for_max_cond(t, eta, cond_eps_norm, max_cond):
        # This returns the h that should be used for the given cond_scale norm to keep
        # the norm of its contribution to a step below max_cond at a given t.
        sigma = t_to_sigma(t)
        h = (cond_eps_norm / (cond_eps_norm - max_cond / sigma)).log() / (eta + 1)
        return h.nan_to_num(nan=float("inf"))

    # Set up constants
    sigma_min = torch.tensor(sigma_min, device=x.device)
    sigma_max = torch.tensor(sigma_max, device=x.device)
    max_h = torch.tensor(max_h, device=x.device)
    s_in = x.new_ones([x.shape[0]])
    t_end = sigma_to_t(sigma_min)

    # Set up state
    t = sigma_to_t(sigma_max)
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None
    i = 0

    # Main loop
    while t < t_end - 1e-5:
        # Call model and cond_fn
        sigma = t_to_sigma(t)
        denoised, cond_score = model(x, sigma * s_in)

        # Scale step size down if cond_score is too large
        cond_eps_norm = cond_score.mul(sigma).pow(2).mean().sqrt() + 1e-8
        h = h_for_max_cond(t, eta, cond_eps_norm, max_cond)
        h = max_h * torch.tanh(h / max_h)
        t_next = torch.minimum(t + h, t_end)
        h = t_next - t
        sigma_next = t_to_sigma(t_next)

        # Callback
        if callback is not None:
            obj = {
                "x": x,
                "i": i,
                "sigma": sigma,
                "sigma_next": sigma_next,
                "denoised": denoised,
            }
            callback(obj)

        # First order step (guided)
        h_eta = h + eta * h
        x = (sigma_next / sigma) * torch.exp(-h * eta) * x
        x = x - phi_1(h_eta) * (denoised + sigma**2 * cond_score)
        noise = noise_sampler(sigma, sigma_next)
        x = x + noise * sigma_next * phi_1(2 * eta * h * s_noise).neg().sqrt()

        # Higher order correction (not guided)
        if solver_type == "dpm3" and denoised_2 is not None:
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = phi_1(h_eta) / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        elif solver_type in {"heun", "dpm3"} and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = phi_1(h_eta) / h_eta + 1
            x = x + phi_2 * d
        elif solver_type == "midpoint" and denoised_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            x = x - 0.5 * phi_1(h_eta) * d

        # Update state
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
        t += h
        i += 1

    return x


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("prompt", type=str, default="", help="the text prompts")
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="the diffusion model checkpoint to load",
    )
    p.add_argument(
        '--config',
        type=Path,
        help='configuration file for inference-only checkpoints')
    p.add_argument(
        "--clip-model",
        type=str,
        nargs="+",
        default=["ViT-B/16"],
        choices=clip.available_models(),
        help="the CLIP model to use",
    )
    p.add_argument(
        "--clip-scale",
        "-cs",
        type=float,
        nargs="+",
        default=[2000.0],
        help="the CLIP guidance scale",
    )
    p.add_argument("--compile", action="store_true", help="torch.compile() the model")
    p.add_argument("--clip_jit", action="store_true", help="download JITed CLIP")
    p.add_argument(
        "--cutn",
        type=int,
        nargs="+",
        default=[32],
        help="the number of random crops to use per step",
    )
    p.add_argument("--device", type=str, default=None, help="the device to use")
    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="the multiplier for the noise variance. 0 gives ODE sampling, 1 gives standard diffusion SDE sampling.",
    )
    p.add_argument(
        "--image-prompts", type=str, nargs="*", default=[], help="the image prompts"
    )
    p.add_argument("--init", type=Path, help="the initial image")
    p.add_argument(
        "--init-sigma",
        type=float,
        default=10.0,
        help="the starting noise level when using an init image",
    )
    p.add_argument(
        "--max-cond",
        type=float,
        default=0.05,
        help="the maximum amount that guidance is allowed to perturb a step",
    )
    p.add_argument(
        "--max-h",
        type=float,
        default=0.1,
        help="the maximum step size",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=["eps", "v"],
        default="eps",
        help="the model type",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=Path("out.png"), help="the output file"
    )
    p.add_argument(
        "--save-all", action="store_true", help="save all intermediate denoised images"
    )
    p.add_argument("--seed", type=int, default=0, help="the random seed")
    p.add_argument(
        "--size", type=int, nargs=2, default=(512, 512), help="the output size"
    )
    p.add_argument(
        "--solver",
        type=str,
        choices=("euler", "midpoint", "heun", "dpm3"),
        default="dpm3",
        help="the SDE solver type",
    )
    args = p.parse_args()

    if not len(args.clip_model) == len(args.clip_scale) == len(args.cutn):
        raise ValueError(
            "--clip-model, --clip-scale, and --cutn must have the same number of arguments"
        )

    print(Panel(Align("CLIP Guided Diffusion", "center")))

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f'Using device: "{device}"')

    torch.set_float32_matmul_precision("high")

    # Load diffusion model
    print("Loading diffusion model.")
    checkpoint = args.checkpoint
    if checkpoint is None and args.config is None:
        checkpoint = download_file(
            url="https://models.rivershavewings.workers.dev/512x512_diffusion_uncond_finetune_008100.safetensors",
            root=Path(torch.hub.get_dir()) / "checkpoints" / "rivershavewings",
            expected_sha256="02e212cbec7c9012eb12cd63fef6fa97640b4e8fcd6c6e1f410a52eea1925fe1",
        )
        config = K.config.load_config(config_dir / "guided_diffusion_kat.json")
    else:
        assert args.config is not None, "--checkpoint requires a corresponding config.json to be passed via --config"
        config = K.config.load_config(args.config)
    model_config = config['model']

    if model_config['type'] == 'guided_diffusion':
        from .load_diffusion_model import construct_diffusion_model, load_diffusion_model, wrap_diffusion_model
        # can't easily put this into K.config.make_model; would change return type and introduce dependency
        gdiff_model, guided_diff = construct_diffusion_model(model_config['config'])
        model_type: Literal['eps', 'v'] = model_config["objective"]
        load_diffusion_model(checkpoint, gdiff_model)
        gdiff_model.requires_grad_(False).eval().to(device)
        model = wrap_diffusion_model(gdiff_model, guided_diff, device=device, model_type=model_type)
        sigma_min, sigma_max = model.sigma_min.item(), model.sigma_max.item()
        class_cond_key = 'y'
    else:
        kdiff_model = K.config.make_model(config)
        ckpt = safetorch.load_file(args.resume_inference)
        kdiff_model.load_state_dict(ckpt)
        kdiff_model.requires_grad_(False).eval().to(device)
        model = K.config.make_denoiser_wrapper(config)(kdiff_model)
        sigma_min, sigma_max = model_config['sigma_min'], model_config['sigma_max']
        class_cond_key = 'class_cond'
    size_fac = (args.size[0] * args.size[1]) / (512 * 512)

    # Load CLIP
    print("Loading CLIP.")
    clip_wraps = [
        CLIPWrapper.from_pretrained(name, device=device, cutn=cutn, jit=bool(args.clip_jit))
        for name, cutn in zip(args.clip_model, args.cutn)
    ]

    # Parse and encode prompts
    prompts, image_prompts, weights, targets = [], [], [], []
    for prompt_and_weight in args.prompt.split("|"):
        a, b, c = prompt_and_weight.rpartition(":")
        if not b:
            a, c = c, "1"
        prompt, weight = a.strip(), float(c.strip())
        if prompt:
            prompts.append(prompt)
            weights.append(weight)
    for prompt_and_weight in args.image_prompts:
        a, b, c = prompt_and_weight.rpartition(":")
        if not b:
            a, c = c, "1"
        prompt, weight = a.strip(), float(c.strip())
        prompt = Image.open(prompt).convert("RGB")
        prompt = TF.to_tensor(prompt).to(device)[None] * 2 - 1
        image_prompts.append(prompt)
        weights.append(weight)
    weights = torch.tensor(weights, device=device)
    for wrap in clip_wraps:
        embeds = list(wrap.encode_text(prompts))
        embeds.extend(wrap.encode_image(ip) for ip in image_prompts)
        embeds = torch.stack(embeds)
        targets.append(spherical_average(embeds, weights))

    # Wrap the model in a function that also computes and returns the cond_score
    def cond_model(x, sigma, **kwargs):
        denoised = None

        def loss_fn(x):
            nonlocal denoised
            denoised = model(x, sigma, **kwargs)
            loss = x.new_tensor(0.0)
            for wrap, target, scale in zip(clip_wraps, targets, args.clip_scale):
                image_embed = wrap(denoised)
                loss_cur = dist(image_embed, target) ** 2 / 2
                loss += loss_cur * scale * size_fac
            return loss

        grad = torch.autograd.functional.vjp(loss_fn, x)[1]
        return denoised.clamp(-1, 1), -grad

    if args.compile:
        cond_model = torch.compile(cond_model)

    save_fn = partial(save_image, prompt=args.prompt, args=args)

    # Set up callback
    class Callback:
        def __enter__(self):
            self.ex = futures.ThreadPoolExecutor()
            self.pbar = tqdm()
            self.steps = 0
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.pbar.close()
            self.ex.shutdown()

        def __call__(self, info):
            self.pbar.update(1)
            self.steps += 1
            i = info["i"]
            sigma = info["sigma"].item()
            sigma_next = info["sigma_next"].item()
            h = math.log(sigma / sigma_next)
            print(f"step {i}, sigma: {sigma:g}, h: {h:g}")
            if args.save_all:
                path = args.output.with_stem(args.output.stem + f"_{i:05}")
                self.ex.submit(save_fn, info["denoised"][0], path, steps=self.steps)

    # Load init image
    if args.init is None:
        init_sigma = sigma_max
        x = torch.zeros([1, 3, args.size[1], args.size[0]], device=device)
    else:
        print("Loading init image.")
        init_sigma = min(max(args.init_sigma, sigma_min), sigma_max)
        init = Image.open(args.init).convert("RGB").resize(args.size, Image.BICUBIC)
        x = TF.to_tensor(init).to(device)[None] * 2 - 1

    # Draw random noise
    torch.manual_seed(args.seed)
    x = x + torch.randn_like(x) * init_sigma
    ns = K.sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)

    with Callback() as cb:
        # Sample
        print("Sampling.")
        try:
            samples = sample_dpm_guided(
                model=cond_model,
                x=x,
                sigma_min=sigma_min,
                sigma_max=init_sigma,
                max_h=args.max_h,
                max_cond=args.max_cond,
                eta=args.eta,
                noise_sampler=ns,
                solver_type=args.solver,
                callback=cb,
            )

            # Save the image
            print(f"Saving to {args.output}...")
            save_fn(samples[0], args.output, steps=cb.steps)
        except KeyboardInterrupt:
            print("Interrupted")


if __name__ == "__main__":
    main()
