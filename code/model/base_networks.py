import numpy as np
import torch.nn as nn
from hashencoder.hashgrid import HashEncoder
from model.embedder import *
from utils import rend_util

class ImplicitNetworkGrid_COMBINE(nn.Module):
    """
    Implicit network with the combination of coarse and fine networks.

    """

    def __init__(self, conf, feature_vector_size, sdf_bounding_sphere):
        super().__init__()
        self.feature_vector_size = feature_vector_size
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.coarse = ImplicitNetworkGrid(
            feature_vector_size, sdf_bounding_sphere, name="coarse", **conf.get_config("coarse")
        )
        self.fine = ImplicitNetworkGrid(
            feature_vector_size, sdf_bounding_sphere, name="fine", **conf.get_config("fine")
        )

    def forward():
        pass

    def get_sdf_vals(self, x, stage="fine"):
        if stage == "coarse":
            return self.coarse.get_sdf_vals(x)
        else:
            c_feature_vectors = self.coarse.get_feature(x)
            return self.coarse.get_sdf_vals(x) + self.fine.get_sdf_vals(x, c_feature_vectors)

    def get_outputs(self, x, stage="fine"):
        if stage == "coarse":
            return self.coarse.get_outputs(x)
        else:
            c_sdf, c_feature_vectors, c_gradients = self.coarse.get_outputs(x)
            f_sdf, f_feature_vectors, f_gradients = self.fine.get_outputs(x, c_feature_vectors=c_feature_vectors)
            return c_sdf + f_sdf, c_feature_vectors + f_feature_vectors, c_gradients + f_gradients

    def gradient(self, x, stage="fine"):
        if stage == "coarse":
            return self.coarse.gradient(x)
        else:
            c_feature_vectors = self.coarse.get_feature(x)
            return self.coarse.gradient(x) + self.fine.gradient(x, c_feature_vectors)


class ImplicitNetworkGrid(nn.Module):
    """
    Feature grid and decoder MLP.

    """

    def __init__(
        self,
        feature_vector_size,
        sdf_bounding_sphere,
        d_in,
        d_out,
        dims,
        geometric_init=True,
        bias=1.0,
        skip_in=(),
        weight_norm=True,
        multires=0,
        sphere_scale=1.0,
        inside_outside=False,
        base_size=16,
        end_size=2048,
        logmap=19,
        num_levels=16,
        level_dim=2,
        embedding_method="nerf",
        divide_factor=1.5,
        use_grid_feature=True,
        name="",
        clamp=False,
        concat_coarse_feature=False,
    ):
        super().__init__()
        self.concat_coarse_feature = concat_coarse_feature
        self.name = name
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale

        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        if self.concat_coarse_feature:
            dims[0] += feature_vector_size
        self.clamp = clamp

        # print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        # print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")

        self.encoding = HashEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=level_dim,
            per_level_scale=2,
            base_resolution=base_size,
            log2_hashmap_size=logmap,
            desired_resolution=end_size,
        )

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, embed_type=embedding_method)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, c_feature_vectors=None):
        if self.use_grid_feature:
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
        if (c_feature_vectors is not None) and (self.concat_coarse_feature):
            feature = torch.cat([feature, c_feature_vectors], dim=-1)
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        if self.clamp:
            if self.name == "fine":
                sdf, feature = x[:, :1], x[:, 1:]
                tanh = nn.Tanh()
                sdf = tanh(sdf) * 0.05
                x = torch.cat([sdf, feature], dim=-1)
        return x

    def get_feature(self, x, c_feature_vectors=None, stage=None):
        if self.concat_coarse_feature:
            feat = self.forward(x, c_feature_vectors)[:, 1:]
        else:
            feat = self.forward(x)[:, 1:]
        return feat

    def gradient(self, x, c_feature_vectors=None, stage=None):
        x.requires_grad_(True)
        if self.concat_coarse_feature:
            y = self.forward(x, c_feature_vectors)[:, :1]
        else:
            y = self.forward(x)[:, :1]

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return gradients

    def get_outputs(self, x, c_feature_vectors=None, stage=None):
        x.requires_grad_(True)
        if self.concat_coarse_feature:
            output = self.forward(x, c_feature_vectors)
        else:
            output = self.forward(x)
        sdf = output[:, :1]
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x, c_feature_vectors=None, stage=None):
        if self.concat_coarse_feature:
            sdf = self.forward(x, c_feature_vectors)[:, :1]
        else:
            sdf = self.forward(x)[:, :1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    """
    Rendering network.

    """

    def __init__(
        self,
        feature_vector_size,
        mode,
        d_in,
        d_out,
        dims,
        weight_norm=True,
        multires_view=0,
        per_image_code=False,
        model_exposure=False,
        n_images=2000,
        embedding_method="nerf",
        use_grid_feature=False,
    ):
        super().__init__()

        self.use_grid_feature = use_grid_feature
        if self.use_grid_feature:
            self.divide_factor = 1.0
            base_size = 16
            end_size = 2048
            logmap = 24
            num_levels = 16
            level_dim = 2
            self.grid_feature_dim = num_levels * level_dim

            # print(f"using color hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
            # print(f"resolution: {base_size} -> {end_size} with hash map size {logmap}")
            self.encoding = HashEncoder(
                input_dim=3,
                num_levels=num_levels,
                level_dim=level_dim,
                per_level_scale=2,
                base_resolution=base_size,
                log2_hashmap_size=logmap,
                desired_resolution=end_size,
            )
        else:
            self.grid_feature_dim = 0

        self.n_images = n_images
        self.mode = mode
        if self.mode == "no_feature" or self.mode == "no_feature_no_noraml":
            feature_vector_size = 0
        dims = [d_in + feature_vector_size + self.grid_feature_dim] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, embed_type=embedding_method)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        self.per_image_code = per_image_code
        if self.per_image_code:
            self.embeddings = nn.Parameter(torch.empty(self.n_images, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        self.model_exposure = model_exposure
        if self.model_exposure:
            # urban radiance field way of modeling exposure
            code_dim = 4
            fea_dim = 64
            self.embeddings = nn.Parameter(torch.empty(self.n_images, code_dim))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            self.net_exp = nn.Sequential(
                nn.Linear(code_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6)
            )

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices, color_stage="base"):
        if self.use_grid_feature:
            if color_stage == "highfreq":
                grid_feature = self.encoding(points / self.divide_factor)
            elif color_stage == "base":
                grid_feature = self.encoding(points / self.divide_factor)
                grid_feature = grid_feature.detach()

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            if self.use_grid_feature:
                rendering_input = torch.cat([points, view_dirs, normals, feature_vectors, grid_feature], dim=-1)
            else:
                rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)

        if self.mode == "idr_detach":
            rendering_input = torch.cat([points, view_dirs, normals.detach(), feature_vectors], dim=-1)
        elif self.mode == "idr_nopts":
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "idr_nopts_detach":
            rendering_input = torch.cat([view_dirs, normals.detach(), feature_vectors], dim=-1)
        elif self.mode == "idr_nonormal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == "idr_noview":
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "nerf":
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        elif self.mode == "no_feature":
            rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
        elif self.mode == "no_feature_no_noraml":
            rendering_input = torch.cat([points, view_dirs], dim=-1)
        elif self.mode == "no_color":
            return self.sigmoid(feature_vectors[:, :3])

        if self.per_image_code:
            image_code = self.embeddings[indices].repeat(rendering_input.shape[0] // indices.shape[0], 1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.model_exposure:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            out = self.net_exp(image_code)
            R = rend_util.from_euler(out[..., :3])
            t = out[..., 3:]
            # handle the exposure change
            x_nor = torch.matmul(R, x[..., None])[..., 0] + t
            x_nor = self.sigmoid(x_nor)
            x = self.sigmoid(x)
            return x_nor, x

        x = self.sigmoid(x)
        return x

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        return self.encoding.parameters()
