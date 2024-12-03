### Sparse Autoencoder code taken and modified from Linus Lee's Prism project

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin

from enum import Enum

class SparseAutoencoderType(Enum):
    BASIC = "basic"
    TOPK = "topk"


class SparseAutoencoderConfig(BaseModel):
    d_model: int
    d_sparse: int
    sparsity_alpha: float = 0.0  # doesn't matter for inference
    sae_type: SparseAutoencoderType = SparseAutoencoderType.BASIC # L1 SAE is the default
    top_k: int = 0 # Only used for TopK sparse autoencoder


class SparseAutoencoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.config = config

        # from https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder
        self.enc_bias = nn.Parameter(torch.zeros(config.d_sparse))
        self.encoder = nn.Linear(config.d_model, config.d_sparse, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(config.d_model))
        self.decoder = nn.Linear(config.d_sparse, config.d_model, bias=False)

    def forward(
        self,
        x: torch.FloatTensor,
        return_loss: bool = False,
        sparsity_scale: float = 1.0,
        new_loss: bool = True,
    ):
        if self.config.sae_type == SparseAutoencoderType.BASIC:
            return self.forward_basic(x, return_loss, sparsity_scale, new_loss)
        elif self.config.sae_type == SparseAutoencoderType.TOPK:
            return self.forward_topk(x)

    def forward_topk(self, x: torch.FloatTensor, return_loss: bool = False):
        pre_acts = self.encode(x)

        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        hidden_acts = self.decode_topk(top_acts, top_indices)
        sae_out = self.decode(hidden_acts)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        reconstruction_loss = F.mse_loss(sae_out, x)

        return (
            sae_out,
            hidden_acts,
            l2_loss,
            reconstruction_loss,
        )

    def forward_basic(
        self,
        x: torch.FloatTensor,
        return_loss: bool = False,
        sparsity_scale: float = 1.0,
        new_loss: bool = True,
    ):
        f = self.encode(x)
        y = self.decode(f)

        if return_loss:
            reconstruction_loss = F.mse_loss(y, x)
            # print(x.shape)

            decoder_norms = torch.norm(self.decoder.weight, dim=0)

            if new_loss:
                sparsity_loss = (
                    sparsity_scale
                    * self.config.sparsity_alpha
                    * (f.abs() @ decoder_norms).sum()
                )  # TODO: change this to the actual loss function
            else:
                sparsity_loss = (
                    sparsity_scale * self.config.sparsity_alpha * (f.abs().sum())
                )

            loss = reconstruction_loss / x.shape[1] + sparsity_loss
            return y, f, loss, reconstruction_loss

        return y, f, None, None

    def select_topk(self, latents: torch.FloatTensor) -> torch.FloatTensor:
        """Select the top-k latents."""
        return latents.topk(self.config.top_k, sorted=False)

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return F.relu(self.encoder(x - self.dec_bias) + self.enc_bias)

    def decode(self, f: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder(f) + self.dec_bias

    def decode_topk(self, top_acts: torch.FloatTensor, top_indices: torch.FloatTensor) -> torch.FloatTensor:
        # assert self.W_dec is not None, "Decoder weight was not initialized."

        W_dec = self.decoder.weight

        buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        return acts

    def load(self, path: os.PathLike, device: torch.device = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


# Pre-trained configs from HF
class PretrainedConfig:
    sm_v6 = SparseAutoencoderConfig(d_model=512, d_sparse=8 * 512)
    bs_v6 = SparseAutoencoderConfig(d_model=768, d_sparse=8 * 768)
    lg_v6 = SparseAutoencoderConfig(d_model=1024, d_sparse=8 * 1024)
    xl_v6 = SparseAutoencoderConfig(d_model=2048, d_sparse=8 * 2048)
