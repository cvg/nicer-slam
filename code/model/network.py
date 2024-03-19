import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import rend_util
from utils.general import uv2patch
from model.density import LaplaceDensity, GridPredefineDensity
from model.ray_sampler import ImportantSampler
from utils.general import index_to_1d
from model.base_networks import ImplicitNetworkGrid_COMBINE, RenderingNetwork


class SLAMNetwork(nn.Module):
    def __init__(self, conf, dataset=None, n_images=2000):
        super().__init__()

        self.dataset = dataset
        self.H, self.W = self.dataset.img_res
        self.white_bkgd = conf.get_bool("white_bkgd", default=False)
        self.feature_vector_size = conf.get_int("feature_vector_size")
        self.use_warp_loss = conf.get_bool("use_warp_loss", default=False)
        self.embedding_method = conf.get_string("embedding_method", default="nerf")
        self.mapping_patchsizes = conf.get_list("mapping_patchsizes", default=[1, 5, 11])
        self.tracking_patchsizes = conf.get_list("tracking_patchsizes", default=[1, 5, 11])
        self.scene_bounding_sphere = conf.get_float("scene_bounding_sphere", default=1.0)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.implicit_network = ImplicitNetworkGrid_COMBINE(
            conf.get_config("implicit_network"),
            self.feature_vector_size,
            0.0 if self.white_bkgd else self.scene_bounding_sphere,
        )
        self.rendering_network = RenderingNetwork(
            self.feature_vector_size,
            n_images=n_images,
            embedding_method=self.embedding_method,
            **conf.get_config("rendering_network"),
        )
        # init the density method
        self.density_method = conf.get_string("density_method", default="volsdf_gridpredefined")
        if self.density_method == "volsdf_laplace":
            self.density = LaplaceDensity(**conf.get_config("density"))
        elif self.density_method == "volsdf_gridpredefined":
            self.density = GridPredefineDensity(**conf.get_config("gridpredefinedensity"))

        # init the sampling method
        sampling_method = conf.get_string("sampling_method", default="important")
        if sampling_method == "important":
            self.ray_sampler = ImportantSampler(self.scene_bounding_sphere, **conf.get_config("ray_sampler"))
        else:
            raise NotImplementedError
        self.sampling_method = sampling_method

        # init the voxel counter
        self.voxel_res = conf.get_int("voxel_res", default=64)
        self.voxels = torch.zeros((self.voxel_res, self.voxel_res, self.voxel_res)).cuda()
        self.voxels_shape = self.voxels.shape
        if "gridpredefined" in self.density_method:
            self.density.voxels = self.voxels
            self.density.voxel_res = self.voxel_res

    def update_voxels(self, x):
        """
        Update the voxel counter.
        """
        mask = torch.zeros((x.shape[0])).bool().cuda()
        for dim in range(3):
            dim_mask = torch.abs(x[:, dim]) > 0.99
            mask |= dim_mask
        x = x[~mask]
        x = (x + 1) / 2
        x = (x * self.voxel_res).long()
        self.voxels = self.voxels.view(-1)
        tmp_ind = index_to_1d(x, self.voxel_res)
        self.voxels.index_add_(0, tmp_ind, torch.ones_like(tmp_ind).float())
        self.voxels = self.voxels.reshape(self.voxels_shape)

    def forward(
        self,
        input,
        indices,
        ground_truth,
        keyframe_list=None,
        frame_idx=-1,
        mode="vis",
        stage="fine",
        color_stage="highfreq",
        iter=0,
    ):
        if mode == "tracking":
            self.patchsizes = self.tracking_patchsizes
        elif mode == "mapping":
            self.patchsizes = self.mapping_patchsizes

        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        tmp_pose = torch.eye(4).to(pose.device)[None].repeat(pose.shape[0], 1, 1)
        # we need to use for unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, tmp_pose, intrinsics)
        depth_scale = ray_dirs_tmp[:, :, 2:]
        bs = depth_scale.shape[0]

        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, frame_idx, keyframe_list, mode)

        N_samples = z_vals.shape[1]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        points_flat = points.reshape(-1, 3)

        if mode == "mapping":
            self.update_voxels(points_flat.clone().detach())

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat, stage=stage)

        def save_grad(grad):
            self.color_normal_grad = grad.detach()

        gradients.register_hook(save_grad)

        rgb_flat = self.rendering_network(
            points_flat, gradients, dirs_flat, feature_vectors, indices, color_stage=color_stage
        )
        if self.rendering_network.model_exposure:
            rgb, rgb_un = rgb_flat[0].reshape(-1, N_samples, 3), rgb_flat[1].reshape(-1, N_samples, 3)
        else:
            rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(
            z_vals,
            sdf,
            points_flat,
            rays_o=cam_loc,
            rays_d=ray_dirs,
            gradients=gradients,
            frame_idx=frame_idx,
            mode=mode,
        )
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) + 1e-8)
        rendered_depth = depth_values.unsqueeze(2)
        points = cam_loc.unsqueeze(1) + rendered_depth * ray_dirs.unsqueeze(1)
        points = points.reshape(bs, -1, 3).permute(0, 2, 1)

        if "edges" in ground_truth:
            idii, idjj, ii, jj = ground_truth["edges"]
            # define the flow ii->jj
            target_pose = pose[idjj]
            # c2w -> w2c
            target_pose = torch.linalg.inv(target_pose)
            target_intrinsics = intrinsics[idjj]
            reference_uv = uv[idii]
            reference_points = points[idii]
            cam_cord_points = target_pose[:, :3, :3] @ reference_points + target_pose[:, :3, 3:]
            tmp = (target_intrinsics[:, :3, :3] @ cam_cord_points).permute(0, 2, 1)
            flow_uv = tmp[..., :2] / (tmp[..., 2:] + 1e-8)
            flow = flow_uv - reference_uv

        if self.use_warp_loss and ("vis" not in mode) and ("tracking" not in mode):
            warp_output = {}
            full_rgb = ground_truth["full_rgb"]
            full_rgb = full_rgb.reshape(batch_size, self.H, self.W, 3)

            full_depth = ground_truth["full_depth"]
            full_depth = full_depth.reshape(batch_size, self.H, self.W, 1)

            warp_rendered_depth = rendered_depth.reshape(batch_size, -1, 1, 1)

            for patchsize in self.patchsizes:
                uv_patch = uv2patch(uv, patchsize)
                uv_patch = uv_patch.reshape(batch_size, -1, 2)
                ray_dirs_patch, cam_loc_patch = rend_util.get_camera_params(uv_patch, pose, intrinsics)
                ray_dirs_patch = ray_dirs_patch.reshape(batch_size, -1, patchsize * patchsize, 3)
                uv_patch_points = cam_loc_patch.unsqueeze(1).unsqueeze(1) + warp_rendered_depth * ray_dirs_patch
                uv_patch_points  # (batch_size, N_pixels, patchsize*patchsize, 3)

                # the permutation is for later R, T projection
                uv_patch_points = uv_patch_points.reshape(-1, 3).permute(1, 0)
                target_pose = pose.clone()
                target_pose = torch.linalg.inv(target_pose)  # (batch_size, 4, 4)

                cam_cord_points = target_pose[:, :3, :3] @ uv_patch_points + target_pose[:, :3, 3:]
                target_intrinsics = intrinsics.clone()
                tmp = (target_intrinsics[:, :3, :3] @ cam_cord_points).permute(0, 2, 1)
                tmp = tmp.reshape(batch_size, batch_size, -1, patchsize * patchsize, 3)
                target_uv = tmp[..., :2] / (
                    tmp[..., 2:] + 1e-8
                )  # (target, reference, N_pixels, patchsize*patchsize, 2)
                target_uv_depth = tmp[..., 2:]
                target_uv[..., 0] = target_uv[..., 0] / self.W
                target_uv[..., 1] = target_uv[..., 1] / self.H
                target_uv = target_uv * 2 - 1.0  # change range to [-1, 1]
                target_uv = target_uv.reshape(batch_size, -1, 1, 2)
                target_uv_depth = target_uv_depth.reshape(batch_size, -1, 1)
                full_rgb_warp = full_rgb.permute(0, 3, 1, 2)  # (batch_size, 3, self.H, self.W)
                sampled_rgb = F.grid_sample(
                    full_rgb_warp, target_uv, mode="bilinear", padding_mode="zeros", align_corners=True
                )
                target_sampled_rgb = sampled_rgb.reshape(batch_size, 3, batch_size, -1, patchsize * patchsize).permute(
                    0, 2, 3, 4, 1
                )
                target_sampled_rgb_mask = (
                    (target_uv[..., 0] > -1)
                    & (target_uv[..., 0] < 1)
                    & (target_uv[..., 1] > -1)
                    & (target_uv[..., 1] < 1)
                    & (target_uv_depth > 0)
                )
                target_sampled_rgb_mask = target_sampled_rgb_mask.reshape(
                    batch_size, batch_size, -1, patchsize * patchsize
                )
                # target_sampled_rgb: (target, reference, N_pixels, patchsize*patchsize, 3)

                # sample rgb from target image
                # uv_patch size: (batch_size, -1, 2)
                gt_warp_rgbs = []
                gt_warp_depths = []
                gt_warp_rgb_masks = []
                for idx in range(batch_size):
                    gt_warp_rgb_mask = (
                        (0 <= uv_patch[idx, :, 0])
                        & (0 <= uv_patch[idx, :, 1])
                        & (uv_patch[idx, :, 0] < self.W)
                        & (uv_patch[idx, :, 1] < self.H)
                    )
                    gt_warp_rgb = torch.ones(uv_patch.shape[1], 3).cuda()
                    gt_warp_depth = torch.ones(uv_patch.shape[1], 1).cuda()
                    tmp = full_rgb[
                        idx, uv_patch[idx, gt_warp_rgb_mask, 1].long(), uv_patch[idx, gt_warp_rgb_mask, 0].long(), :
                    ]
                    gt_warp_rgb[gt_warp_rgb_mask] = tmp
                    gt_warp_rgbs.append(gt_warp_rgb)
                    gt_warp_rgb_masks.append(gt_warp_rgb_mask)
                    tmp = full_depth[
                        idx, uv_patch[idx, gt_warp_rgb_mask, 1].long(), uv_patch[idx, gt_warp_rgb_mask, 0].long(), :
                    ]
                    gt_warp_depth[gt_warp_rgb_mask] = tmp
                    gt_warp_depths.append(gt_warp_depth)

                gt_warp_depths = torch.stack(gt_warp_depths, dim=0)
                gt_warp_rgbs = torch.stack(gt_warp_rgbs, dim=0)
                gt_warp_rgb_masks = torch.stack(gt_warp_rgb_masks, dim=0)
                gt_warp_rgb_masks = (
                    gt_warp_rgb_masks.unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                    .reshape(batch_size, batch_size, -1, patchsize * patchsize)
                )
                gt_warp_rgbs = gt_warp_rgbs.reshape(1, batch_size, -1, patchsize * patchsize, 3)
                gt_warp_rgbs = gt_warp_rgbs.repeat(batch_size, 1, 1, 1, 1)
                total_warp_mask = gt_warp_rgb_masks & target_sampled_rgb_mask

                if patchsize > 1:
                    gt_warp_depths = gt_warp_depths.reshape(batch_size, -1, patchsize, patchsize, 1)
                    gt_warp_depths = gt_warp_depths.reshape(batch_size, -1, patchsize * patchsize)
                    gt_warp_depths_mask = torch.var(gt_warp_depths, dim=-1, unbiased=False) < 0.01
                    gt_warp_depths_mask_ray_level = gt_warp_depths_mask.reshape(-1)
                    gt_warp_depths_mask = (
                        gt_warp_depths_mask.unsqueeze(0)
                        .unsqueeze(-1)
                        .repeat(batch_size, 1, 1, patchsize * patchsize)
                        .reshape(batch_size, batch_size, -1, patchsize * patchsize)
                    )
                    total_warp_mask = total_warp_mask & gt_warp_depths_mask
                else:
                    gt_warp_depths_mask_ray_level = None
                warp_output[patchsize] = (
                    gt_warp_rgbs,
                    target_sampled_rgb,
                    total_warp_mask,
                    gt_warp_depths_mask_ray_level,
                )

        depth_values = depth_values.reshape(bs, -1, 1)
        depth_values = depth_scale * depth_values

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1.0 - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        rgb_values = rgb_values.reshape(bs, -1, 3)
        output = {
            "rgb": rgb,
            "rgb_values": rgb_values,
            "depth_values": depth_values,
            "z_vals": z_vals,
            "depth_vals": z_vals * depth_scale.reshape(-1, 1),
            "sdf": sdf.reshape(z_vals.shape),
            "weights": weights,
            "entropy": (-weights * torch.log(weights + 1e-4)).sum(dim=-1).mean(),
            "scene_bounding_sphere": self.scene_bounding_sphere,
        }

        if "flow" in locals().keys():
            output["flow"] = flow

        if "warp_output" in locals().keys():
            output["warp_output"] = warp_output

        if self.rendering_network.model_exposure:
            rgb_un_values = torch.sum(weights.unsqueeze(-1) * rgb_un, 1)
            output["rgb_un"] = rgb_un
            output["rgb_un_values"] = rgb_un_values

        if self.training and ("vis" not in mode) and ("mapping" in mode):

            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels

            eikonal_points = (
                torch.empty(n_eik_points * 10, 3)
                .uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
                .cuda()
            )
            # add some of the near surface points
            with torch.no_grad():
                eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(
                    -1, 3
                )
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points, stage=stage)

            # split gradient to eikonal points and heighbour ponits
            output["grad_theta"] = grad_theta[: grad_theta.shape[0] // 2]
            output["grad_theta_nei"] = grad_theta[grad_theta.shape[0] // 2 :]

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        normal_map = normal_map.reshape(bs, -1, 3)
        rot = pose[:, :3, :3]
        normal_map = torch.einsum("bij,bni->bnj", rot, normal_map)
        output["normal_map"] = normal_map

        return output

    def volume_rendering(
        self, z_vals, sdf, points_flat, rays_o=None, rays_d=None, gradients=None, frame_idx=1, mode=None
    ):

        density_flat = self.density(sdf, x=points_flat)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat(
            [torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1
        )  # shift one step
        alpha = -torch.exp(-free_energy) + 1  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights
