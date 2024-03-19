import abc
import torch
from utils import rend_util


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1):
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75 if far == -1 else far)
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15)  # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far

    def get_z_vals(self, ray_dirs, cam_loc, model):
        ray_dirs = ray_dirs.detach()
        cam_loc = cam_loc.detach()
        if not self.take_sphere_intersection:
            near, far = (
                self.near * torch.ones(ray_dirs.shape[0], 1).cuda(),
                self.far * torch.ones(ray_dirs.shape[0], 1).cuda(),
            )
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).cuda()
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()
            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far


class ImportantSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        inverse_sphere_bg=False,
        N_samples_inverse_sphere=0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=True
        )

        self.N_samples_extra = N_samples_extra

        self.scene_bounding_sphere = scene_bounding_sphere

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model, frame_idx, keyframe_list, mode):
        # Start with uniform sampling
        z_vals, near, far = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None
        dists = z_vals[:, 1:] - z_vals[:, :-1]

        points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)

        points_flat = points.reshape(-1, 3)

        # Calculating the SDF only for the new sampled points
        with torch.no_grad():
            sdf = model.implicit_network.get_sdf_vals(points_flat)

        # Upsample more points
        density = model.density(sdf, x=points_flat).reshape(z_vals.shape)

        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here

        N = self.N_samples

        bins = z_vals
        pdf = weights[..., :-1]
        pdf = pdf + 1e-5  # prevent nans
        pdf = pdf / torch.sum(pdf, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Invert CDF
        u = torch.linspace(0.0, 1.0, steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_samples = samples

        if self.inverse_sphere_bg:  # if inverse sphere then need to add the far sphere intersection
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik
