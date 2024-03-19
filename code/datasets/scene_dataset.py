import os
import torch
import numpy as np
from utils import rend_util
from glob import glob
import cv2
from easydict import EasyDict as edict
import lzma
from utils.cam_util import prealign_cameras_apply_another


class SLAMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
        use_mask=False,
        use_gt_depth=False,
        keyframe_every=10,
        conf=None,
        n_images=2000,
        gt_depth_png_scale=6553.5,
    ):
        self.est_pose_all = {}
        self.sampling_idx = None
        self.conf = conf
        self.scan_id = scan_id
        self.data_dir = data_dir
        self.img_res = img_res
        self.H, self.W = img_res
        self.n_images = n_images
        self.keyframe_every = keyframe_every
        self.gt_depth_png_scale = gt_depth_png_scale
        self.Hedge = self.conf.get_int("SLAM.tracking.Hedge")
        self.Wedge = self.conf.get_int("SLAM.tracking.Wedge")

        self.total_pixels = img_res[0] * img_res[1]
        self.tracking_total_pixels = (img_res[0] - 2 * self.Hedge) * (img_res[1] - 2 * self.Wedge)

        self.instance_dir = os.path.join(data_dir, "scan{0}".format(scan_id))
        if not os.path.exists(self.instance_dir):
            raise FileNotFoundError(f"Data directory is empty !!!!!!")

        # load intrinsics and poses
        self.cam_file = "{0}/cameras.npz".format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict["scale_mat_%d" % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict["world_mat_%d" % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scene_scale = self.get_scale_mat()[0, 0]
        self.gt_pose_all = []
        self.intrinsics_all = []
        for idx, (scale_mat, world_mat) in enumerate(zip(scale_mats, world_mats)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # if intrinsics or pose have nan or inf value, then use the first frame's intrinsics or pose (Happens in ScanNet)
            intrinsics = torch.from_numpy(intrinsics).float()
            if torch.isnan(intrinsics).any() or torch.isinf(intrinsics).any():
                intrinsics = self.intrinsics_all[0]

            self.intrinsics_all.append(intrinsics)
            pose = torch.from_numpy(pose).float()
            if torch.isnan(pose).any() or torch.isinf(pose).any():
                pose = torch.eye(4)

            self.gt_pose_all.append(pose)

        # load images
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        self.image_paths = (
            glob_data(os.path.join("{0}".format(self.instance_dir), "*_rgb.png"))[: self.n_images]
            + glob_data(os.path.join("{0}".format(self.instance_dir), "*_rgb.jpg"))[: self.n_images]
        )
        self.depth_paths = glob_data(os.path.join("{0}".format(self.instance_dir), "*_depth.npy"))[: self.n_images]
        self.normal_paths = glob_data(os.path.join("{0}".format(self.instance_dir), "*_normal.npy"))[: self.n_images]
        if len(self.depth_paths) == 0:
            self.depth_paths = None
        if len(self.normal_paths) == 0:
            self.normal_paths = None
        if use_mask:
            self.mask_paths = glob_data(os.path.join("{0}".format(self.instance_dir), "*_mask.npy"))[: self.n_images]
        else:
            self.mask_paths = None
        if use_gt_depth:
            self.gt_depth_paths = glob_data(os.path.join("{0}".format(self.instance_dir), "*_gt_depth.png"))[
                : self.n_images
            ]
        else:
            self.gt_depth_paths = None

        self.rgb_images = {}
        self.mask_images = {}
        self.depth_images = {}
        self.normal_images = {}
        self.gt_depth_images = {}

        # prepare uv for later random sampling
        uv = np.mgrid[0 : self.img_res[0], 0 : self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        if (self.sampling_idx is not None) and (self.mode == "tracking"):
            uv = uv[:, self.Hedge : self.img_res[0] - self.Hedge, self.Wedge : self.img_res[1] - self.Wedge]
        uv = uv.reshape(2, -1).transpose(1, 0)
        self.uv = uv

    def __len__(self):
        return self.n_images

    def clean(self, idx):
        # print("CLEAN FRAME: ", idx)
        if idx in self.rgb_images:
            del self.rgb_images[idx]
        if idx in self.normal_images:
            del self.normal_images[idx]
        if idx in self.depth_images:
            del self.depth_images[idx]
        if idx in self.mask_images:
            del self.mask_images[idx]
        if idx in self.gt_depth_images:
            del self.gt_depth_images[idx]

    def get_rgb_image(self, idx):
        if idx in self.rgb_images:
            return self.rgb_images[idx]
        else:
            # print("READING FROM DISK, Frame:", idx)
            path = self.image_paths[idx]
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            rgb = torch.from_numpy(rgb).float()
            self.rgb_images[idx] = rgb
            return rgb

    def get_normal_image(self, idx):
        if idx in self.normal_images:
            return self.normal_images[idx]
        else:
            if self.normal_paths is None:
                normal = torch.ones_like(self.get_rgb_image(idx))
            else:
                path = self.normal_paths[idx]
                try:
                    with lzma.open(path, "rb") as f:
                        normal = np.load(f, allow_pickle=True)
                except:
                    normal = np.load(path, allow_pickle=True)
                normal = normal.reshape(3, -1).transpose(1, 0)
                # important as the output of omnidata is normalized
                normal = normal * 2.0 - 1.0
                normal = torch.from_numpy(normal).float()
            self.normal_images[idx] = normal
            return normal

    def get_depth_image(self, idx):
        if idx in self.depth_images:
            return self.depth_images[idx]
        else:
            if self.normal_paths is None:
                depth = torch.ones_like(self.get_rgb_image(idx)[:, 0])
            else:
                path = self.depth_paths[idx]
                try:
                    with lzma.open(path, "rb") as f:
                        depth = np.load(f, allow_pickle=True)
                except:
                    depth = np.load(path, allow_pickle=True)
                depth = torch.from_numpy(depth.reshape(-1, 1)).float()
            self.depth_images[idx] = depth
            return depth

    def get_mask_image(self, idx):
        if idx in self.mask_images:
            return self.mask_images[idx]
        else:
            if self.mask_paths is None:
                if ("Replica" in self.data_dir) and (self.scan_id == 4):
                    # NOTE: add ignore idx for office 4 here
                    ignore_idx = list(range(0, 300)) + list(range(700, 1400)) + list(range(1750, 2000))
                    if idx in ignore_idx:
                        mask = torch.zeros_like(self.get_depth_image(idx))
                    else:
                        mask = torch.ones_like(self.get_depth_image(idx))
                else:
                    mask = torch.ones_like(self.get_depth_image(idx))
            else:
                path = self.mask_paths[idx]
                mask = np.load(path)
                mask = torch.from_numpy(mask.reshape(-1, 1)).float()
            self.mask_images[idx] = mask

            return mask

    def get_gt_depth_image(self, idx):
        if idx in self.gt_depth_images:
            return self.gt_depth_images[idx]
        else:
            if self.gt_depth_paths is None:
                gt_depth = torch.ones_like(self.get_depth_image(idx))
            else:
                path = self.gt_depth_paths[idx]
                gt_depth = cv2.imread(path, -1)
                gt_depth = gt_depth / self.gt_depth_png_scale
                gt_depth = torch.from_numpy(gt_depth.reshape(-1, 1)).float()
            self.gt_depth_images[idx] = gt_depth
            return gt_depth

    def __getitem__(self, idx):
        uv = self.uv
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.est_pose_all[idx],
        }
        rgb_image = self.get_rgb_image(idx)
        mask_image = self.get_mask_image(idx)
        depth_image = self.get_depth_image(idx)
        normal_image = self.get_normal_image(idx)
        gt_depth_image = self.get_gt_depth_image(idx)

        uv = uv.cuda()
        rgb_image = rgb_image.cuda()
        mask_image = mask_image.cuda()
        depth_image = depth_image.cuda()
        normal_image = normal_image.cuda()
        gt_depth_image = gt_depth_image.cuda()

        #  self.sampling_idx is asyncly updated in change_sampling_idx()
        if self.sampling_idx is not None:
            ground_truth = {}
            ground_truth["full_rgb"] = rgb_image
            ground_truth["rgb"] = rgb_image[self.sampling_idx, :]
            ground_truth["mask"] = mask_image[self.sampling_idx, :]
            ground_truth["depth"] = depth_image[self.sampling_idx, :]
            ground_truth["normal"] = normal_image[self.sampling_idx, :]
            ground_truth["full_depth"] = gt_depth_image / self.scene_scale
            ground_truth["gt_depth"] = gt_depth_image[self.sampling_idx, :] / self.scene_scale
            sample["uv"] = uv[self.sampling_idx, :]
            sample["sampling_idx"] = self.sampling_idx
        else:
            # for visualization
            ground_truth = {
                "rgb": rgb_image,
                "mask": mask_image,
                "depth": depth_image,
                "normal": normal_image,
                "gt_depth": gt_depth_image / self.scene_scale,
            }
        for key, val in sample.items():
            sample[key] = val.cuda()
        for key, val in ground_truth.items():
            ground_truth[key] = val.cuda()
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)
        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            # for visualization
            self.sampling_idx = None
        else:
            if self.mode == "tracking":
                total_pixels = self.tracking_total_pixels
            else:
                total_pixels = self.total_pixels
            self.sampling_idx = torch.randint(total_pixels, (sampling_size,))
            self.sampling_idx = self.sampling_idx.cuda()

    def get_scale_mat(self):
        return np.load(self.cam_file)["scale_mat_0"]


class SLAMDataset_EVAL(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
        use_mask=False,
        use_gt_depth=False,
        checkpoints_path=None,
        eval_method=None,
        **kwargs,
    ):
        self.n_images = kwargs["n_images"]
        
        if eval_method == "extrapolate":
            self.idxs = range(100)
        elif eval_method == "interpolate":
            self.idxs = range(2, self.n_images, 100)
        self.est_pose_all = {}
        self.instance_dir = os.path.join(data_dir, "scan{0}".format(scan_id))
        if not os.path.exists(self.instance_dir):
            raise FileNotFoundError(f"Data directory is empty !!!!!!")

        self.mode = ""

        self.img_res = img_res

        assert os.path.exists(self.instance_dir), f"{self.instance_dir} Data directory is empty"

        self.sampling_idx = None

        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        self.cam_file = "{0}/cameras.npz".format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict["scale_mat_%d" % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict["world_mat_%d" % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.gt_pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.gt_pose_all.append(torch.from_numpy(pose).float())

        if True:
            self.gt_pose_all = torch.stack(self.gt_pose_all)[:, :3, :4].cuda()
            scale = 1
            ckptsdir = f"{checkpoints_path}/PoseParameters"
            if os.path.exists(ckptsdir):
                ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if "pth" in f]
                if len(ckpts) > 0:
                    ckpt_path = ckpts[-1]
                    print("Get ckpt :", ckpt_path)
                    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
                    estimate_c2w_list = ckpt["est_pose_all"]
                    gt_c2w_list = ckpt["gt_pose_all"]
                    gt_c2w_list = torch.stack(gt_c2w_list)

                    estimate_c2w_list = list(estimate_c2w_list.values())
                    estimate_c2w_list = torch.stack(estimate_c2w_list)

                    gt_c2w_list = gt_c2w_list.cuda()
                    estimate_c2w_list = estimate_c2w_list.cuda()

                    N = estimate_c2w_list.shape[0]
                    gt_c2w_list = gt_c2w_list[:N][:, :3, :4]
                    estimate_c2w_list = estimate_c2w_list[:N][:, :3, :4]

                    pose_aligned, _ = prealign_cameras_apply_another(gt_c2w_list, estimate_c2w_list, self.gt_pose_all)
                    self.gt_pose_all = pose_aligned.cpu()

        self.image_paths = (
            glob_data(os.path.join("{0}".format(self.instance_dir), "*_rgb.png"))[: self.n_images]
            + glob_data(os.path.join("{0}".format(self.instance_dir), "*_rgb.jpg"))[: self.n_images]
        )

        self.rgb_images = {}

        uv = np.mgrid[0 : self.img_res[0], 0 : self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        self.uv = uv

    def __len__(self):
        return len(self.idxs)

    def clean(self, idx):
        if idx in self.rgb_images:
            del self.rgb_images[idx]

    def get_rgb_image(self, idx):
        if idx in self.rgb_images:
            return self.rgb_images[idx]
        else:
            path = self.image_paths[idx]
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            rgb = torch.from_numpy(rgb).float()
            self.rgb_images[idx] = rgb
            return rgb

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        sample = {
            "uv": self.uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.gt_pose_all[idx],
        }

        rgb_image = self.get_rgb_image(idx)
        ground_truth = {
            "rgb": rgb_image,
        }
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        self.sampling_idx = None

    def get_scale_mat(self):
        return np.load(self.cam_file)["scale_mat_0"]
