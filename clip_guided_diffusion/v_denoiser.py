import torch
import k_diffusion as K
from guided_diffusion.unet import UNetModel
from guided_diffusion.respace import SpacedDiffusion

class OpenAIVDenoiser(K.external.DiscreteVDDPMDenoiser):
    """A wrapper for OpenAI v objective diffusion models."""
    inner_model: UNetModel

    def __init__(
        self, model: UNetModel, diffusion: SpacedDiffusion, quantize=False, has_learned_sigmas=True, device="cpu"
    ):
        alphas_cumprod = torch.tensor(
            diffusion.alphas_cumprod, device=device, dtype=torch.float32
        )
        super().__init__(model, alphas_cumprod, quantize=quantize)
        self.has_learned_sigmas = has_learned_sigmas

    def get_v(self, *args, **kwargs):
        model_output = self.inner_model(*args, **kwargs)
        if self.has_learned_sigmas:
            return model_output.chunk(2, dim=1)[0]
        return model_output