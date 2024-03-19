import torch
from torch import nn
import utils.general as utils
from pytorch_msssim import SSIM
from utils.MiDaS import ScaleAndShiftInvariantLoss


class SLAMLoss(nn.Module):
    def __init__(
        self,
        rgb_loss,
        eikonal_weight,
        trainer=None,
        train_dataset=None,
        assign_scale_shift_init=False,
        smooth_weight=0.005,
        warp_loss_type="l1",
        depth_weight=0.1,
        normal_l1_weight=0.05,
        normal_cos_weight=0.05,
        gt_depth_weight=0.0,
        flow_weight=0.0,
        warp_loss_weight=0,
        scan_id=-1,
        model=None,
        rgb_loss_weight=1.0,
        assign_scale=20.0,
    ):
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.scan_id = scan_id
        self.flow_weight = flow_weight
        self.assign_scale = assign_scale
        self.depth_weight = depth_weight
        self.train_dataset = train_dataset
        self.smooth_weight = smooth_weight
        self.warp_loss_type = warp_loss_type
        self.eikonal_weight = eikonal_weight
        self.rgb_loss_weight = rgb_loss_weight
        self.gt_depth_weight = gt_depth_weight
        self.warp_loss_weight = warp_loss_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.assign_scale_shift_init = assign_scale_shift_init

        self.rgb_loss = utils.get_class(rgb_loss)(reduction="mean")
        self.flow_loss = nn.L1Loss(reduction="mean")
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        if self.warp_loss_type == "ssim":
            self.ssimloss_dict = {}
            self.ssimloss_dict[11] = SSIM(data_range=1, win_size=11, size_average=True, channel=3)
            self.ssimloss_dict[5] = SSIM(data_range=1, win_size=5, size_average=True, channel=3)
            self.ssimloss_dict[3] = SSIM(data_range=1, win_size=3, size_average=True, channel=3)

    def get_rgb_loss(self, rgb_values, rgb_gt, mask=None):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_values = rgb_values.reshape(-1, 3)
        if mask is not None:
            mask = mask.reshape(-1)
            rgb_loss = self.rgb_loss(rgb_values[mask], rgb_gt[mask])
        else:
            rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_gt_depth_loss(self, depth_values, depth_gt, mask=None):
        depth_gt = depth_gt.reshape(-1, 1)
        depth_values = depth_values.reshape(-1, 1)
        if mask is not None:
            mask = mask.reshape(-1)
            depth_loss = torch.abs(depth_values[mask] - depth_gt[mask]).mean()
        else:
            depth_loss = torch.abs(depth_values - depth_gt).mean()
        return depth_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self, model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs["grad_theta"]
        g2 = model_outputs["grad_theta_nei"]
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss = torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss

    def get_depth_loss(self, depth_pred, depth_gt, mask, keyframe_list):
        return self.depth_loss(depth_pred, (depth_gt * 50 + 0.5), mask, keyframe_list)

    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
        return l1, cos

    def get_flow_loss(self, model_outputs, ground_truth, keyframe_list):
        if "flow" not in model_outputs:
            return 0.0
        if keyframe_list is None:
            loss = self.flow_loss(
                model_outputs["flow"][ground_truth["flow_mask"]], ground_truth["flow"][ground_truth["flow_mask"]].cuda()
            )
        else:
            flow_mask = ground_truth["flow_mask"]
            loss = self.flow_loss(model_outputs["flow"][flow_mask], ground_truth["flow"][flow_mask].cuda())

        return loss

    def forward(
        self,
        model_outputs,
        ground_truth,
        keyframe_list=None,
        frame_idx=0,
        stage="coarse",
    ):
        rgb_gt = ground_truth["rgb"].cuda()
        depth_gt = ground_truth["depth"].cuda()
        normal_gt = ground_truth["normal"].cuda()
        depth_real_gt = ground_truth["gt_depth"].cuda()

        rgb_pred = model_outputs["rgb_values"]
        depth_pred = model_outputs["depth_values"]
        normal_pred = model_outputs["normal_map"][None]
        bs = depth_pred.shape[0]

        rgb_loss = self.get_rgb_loss(rgb_pred, rgb_gt)

        if ("warp_output" in model_outputs) and self.warp_loss_weight > 0 and stage == "fine" and (frame_idx != 0):
            warp_loss = 0.0
            warp_output = model_outputs["warp_output"]
            for patchsize in warp_output.keys():
                # target_sampled_rgb is the one after grid_sample
                gt_warp_rgbs, target_sampled_rgb, total_warp_mask, gt_warp_depths_mask_ray_level = warp_output[
                    patchsize
                ]
                if (patchsize == 1) or (self.warp_loss_type == "l1"):
                    cur_patchsize_loss = (
                        (target_sampled_rgb[total_warp_mask] - gt_warp_rgbs[total_warp_mask]).abs().mean()
                    )
                elif self.warp_loss_type == "ssim":
                    target_sampled_rgb[~total_warp_mask] = 0.0
                    gt_warp_rgbs[~total_warp_mask] = 0.0
                    target_sampled_rgb = target_sampled_rgb.reshape(-1, patchsize, patchsize, 3).permute(0, 3, 1, 2)
                    gt_warp_rgbs = gt_warp_rgbs.reshape(-1, patchsize, patchsize, 3).permute(0, 3, 1, 2)
                    loss_ssim = self.ssimloss_dict[patchsize](target_sampled_rgb, gt_warp_rgbs)
                    cur_patchsize_loss = 1 - loss_ssim
                    cur_patchsize_loss *= 0.05
                else:
                    raise NotImplementedError("Strange patch loss type")
                warp_loss += cur_patchsize_loss
        else:
            warp_loss = 0.0

        if self.eikonal_weight > 0 and "grad_theta" in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs["grad_theta"])
        else:
            eikonal_loss = 0.0

        # only supervise the foreground normal
        mask = ((model_outputs["sdf"] > 0.0).any(dim=-1) & (model_outputs["sdf"] < 0.0).any(dim=-1))[None, :, None]
        mask = mask.reshape(bs, -1, 1)
        mask = (ground_truth["mask"] > 0.5).cuda() & mask

        if self.depth_weight > 0:
            if ("Replica" in self.train_dataset.data_dir) and (self.scan_id == 4):
                depth_mask = torch.ones_like(depth_pred)
            else:
                depth_mask = mask
            depth_loss = self.get_depth_loss(depth_pred, depth_gt, depth_mask, keyframe_list)
        else:
            depth_loss = 0.0

        # directly assign shift scale for the first frame for init
        if self.assign_scale_shift_init:
            if frame_idx == 0:
                depth_real_gt = depth_gt * self.assign_scale
                self.gt_depth_weight = 10
            else:
                self.gt_depth_weight = 0

        if self.gt_depth_weight > 0:
            gt_depth_mask = (ground_truth["gt_depth"] > 0).cuda()
            gt_depth_loss = self.get_gt_depth_loss(depth_pred, depth_real_gt, gt_depth_mask)
        else:
            gt_depth_loss = 0.0

        if self.normal_l1_weight > 0 or self.normal_cos_weight > 0:
            normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt * mask)
        else:
            normal_l1 = 0.0
            normal_cos = 0.0

        if self.smooth_weight > 0.0:
            smooth_loss = self.get_smooth_loss(model_outputs)
        else:
            smooth_loss = 0.0

        if self.flow_weight > 0.0:
            flow_loss = self.get_flow_loss(model_outputs, ground_truth, keyframe_list)
        else:
            flow_loss = 0.0

        loss = (
            self.flow_weight * flow_loss
            + self.depth_weight * depth_loss
            + self.rgb_loss_weight * rgb_loss
            + self.smooth_weight * smooth_loss
            + self.normal_l1_weight * normal_l1
            + self.warp_loss_weight * warp_loss
            + self.eikonal_weight * eikonal_loss
            + self.normal_cos_weight * normal_cos
            + self.gt_depth_weight * gt_depth_loss
        )

        output = {
            "loss": loss,
            "normal_l1": normal_l1,
            "depth_loss": depth_loss,
            "normal_cos": normal_cos,
            "gt_depth_loss": gt_depth_loss,
            "flow_loss": self.flow_weight * flow_loss,
            "rgb_loss": self.rgb_loss_weight * rgb_loss,
            "warp_loss": self.warp_loss_weight * warp_loss,
            "smooth_loss": self.smooth_weight * smooth_loss,
            "eikonal_loss": self.eikonal_weight * eikonal_loss,
        }

        return output
