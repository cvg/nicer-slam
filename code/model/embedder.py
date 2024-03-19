import torch
import torch.nn as nn


class Embedder:
    """Positional encoding embedding. Code was taken from https://github.com/bmild/nerf."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True, include_input=True):
        super().__init__()
        self.include_input = include_input
        if learnable:
            self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

        self.out_dim = mapping_size
        if include_input:
            self.out_dim += 3

    def forward(self, x):
        x = x.squeeze(0)
        input_x = x
        assert x.dim() == 2, "Expected 2D input (got {}D input)".format(x.dim())
        x = x @ self._B.to(x.device)
        if self.include_input:
            return torch.cat([input_x, torch.sin(x)], -1)
        else:
            return torch.sin(x)


def get_embedder(multires, input_dims=3, embed_type="nerf"):

    if embed_type == "nerf":
        embed_kwargs = {
            "include_input": True,
            "input_dims": input_dims,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)

        def embed(x, eo=embedder_obj):
            return eo.embed(x)

        return embed, embedder_obj.out_dim
    elif embed_type == "fourier":
        embedder_obj = GaussianFourierFeatureTransform(3)

        def embed(x, eo=embedder_obj):
            return eo(x)

        return embed, embedder_obj.out_dim
