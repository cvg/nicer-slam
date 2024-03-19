import os
import sys
import cv2
import lzma
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from pyhocon import ConfigFactory
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import utils.plots as plt
from utils import rend_util
import utils.general as utils
from utils.MiDaS import compute_scale_and_shift
from utils.general import get_tensor_from_camera, get_camera_from_tensor, get_error_degrees

torch.autograd.set_detect_anomaly(True)


class SLAMRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.kwargs = kwargs
        self.is_continue = kwargs["is_continue"]
        self.conf = ConfigFactory.parse_file(kwargs["conf"])
        self.n_images = self.conf.get_int("dataset.n_images")
        self.mapping_window_size = self.conf.get_int("SLAM.mapping.mapping_window_size")
        self.exps_folder_name = kwargs["exps_folder_name"]
        self.const_speed_assumption = self.conf.get_bool("SLAM.tracking.const_speed_assumption", default=False)

        scan_id = kwargs["scan_id"] if kwargs["scan_id"] != -1 else self.conf.get_int("dataset.scan_id", default=-1)
        self.density_method = self.conf.get_string("model.density_method", default="volsdf_gridpredefined")
        self.scan_id = scan_id
        self.mapping_inner_freq = self.conf.get_int("SLAM.mapping.inner_freq", default=1000)
        self.verbose = self.conf.get_bool("SLAM.verbose", default=False)
        self.data_dir = self.conf.get_string("dataset.data_dir")
        self.use_warp_loss = self.conf.get_bool("model.use_warp_loss", default=False)
        self.tracking_change_pix_within_iters = self.conf.get_bool(
            "SLAM.tracking.tracking_change_pix_within_iters", default=True
        )

        self.flow_dir = f"{self.data_dir}/scan{self.scan_id}_pair"
        self.expname = self.conf.get_string("train.expname") + kwargs["expname"]
        if self.scan_id != -1:
            self.expname = self.expname + "_{0}".format(self.scan_id)

        if kwargs["is_continue"] and kwargs["timestamp"] == "latest":
            if os.path.exists(os.path.join("../", kwargs["exps_folder_name"], self.expname)):
                timestamps = os.listdir(os.path.join("../", kwargs["exps_folder_name"], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs["timestamp"]
            is_continue = kwargs["is_continue"]

        utils.mkdir_ifnotexists(os.path.join("../", self.exps_folder_name))
        self.expdir = os.path.join("../", self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        self.timestamp += self.conf.get_string("train.folder_suffix")
        new_expfolder = kwargs["new_expfolder"]
        if is_continue and (not new_expfolder):
            self.timestamp = timestamp
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, "vis")
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, "checkpoints")
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.pose_params_subdir = "PoseParameters"
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.pose_params_subdir))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs["conf"], os.path.join(self.expdir, self.timestamp, "runconf.conf"))
        )

        print("shell command : {0}".format(" ".join(sys.argv)))

        print("Loading data ...")

        dataset_conf = self.conf.get_config("dataset")
        dataset_conf["scan_id"] = self.scan_id
        self.keyframe_every = self.conf.get_int("SLAM.mapping.keyframe_every")
        self.train_dataset = utils.get_class(self.conf.get_string("train.dataset_class"))(
            keyframe_every=self.keyframe_every, conf=self.conf, **dataset_conf
        )

        print("Finish loading data.")

        # init model
        conf_model = self.conf.get_config("model")
        self.model = utils.get_class(self.conf.get_string("train.model_class"))(
            conf=conf_model, dataset=self.train_dataset, n_images=self.n_images
        )
        self.model.train_dataset = self.train_dataset
        self.model.keyframe_every = self.conf.get_int("SLAM.mapping.keyframe_every")
        self.model.cuda()

        # mapping and tracking loss
        self.loss = utils.get_class(self.conf.get_string("train.loss_class"))(
            trainer=self,
            train_dataset=self.train_dataset,
            scan_id=scan_id,
            model=self.model,
            **self.conf.get_config("loss"),
        )
        self.tracking_loss = utils.get_class(self.conf.get_string("train.loss_class"))(
            trainer=self,
            train_dataset=self.train_dataset,
            scan_id=scan_id,
            model=self.model,
            **self.conf.get_config("tracking_loss"),
        )

        # set learning rate
        self.lr = self.conf.get_float("train.learning_rate")
        self.learning_rate_beta = self.conf.get_float("train.learning_rate_beta", default=2.0e-3)
        self.lr_factor_for_fine_grid = self.conf.get_float("train.lr_factor_for_fine_grid", default=1.0)
        self.lr_factor_for_coarse_grid = self.conf.get_float("train.lr_factor_for_coarse_grid", default=1.0)
        self.lr_factor_for_color_grid = self.conf.get_float("train.lr_factor_for_color_grid", default=1.0)

        # load fine level MLP
        pth_path = "pretrain.pth"
        saved_model_state = torch.load(pth_path)
        orig = self.model.state_dict()
        for key, val in saved_model_state["model_state_dict"].items():
            # Fine level MLP
            if ("fine" in key) and ("encoding" not in key):
                orig[key] = val
        self.model.load_state_dict(orig, strict=False)

        # init optimizer
        para_list = [
            {
                "name": "encoding",
                "params": list(self.model.implicit_network.fine.grid_parameters()),
                "lr": self.lr * self.lr_factor_for_fine_grid,
            },
            {
                "name": "encoding",
                "params": list(self.model.implicit_network.coarse.grid_parameters()),
                "lr": self.lr * self.lr_factor_for_coarse_grid,
            },
            {
                "name": "net",
                "params": list(self.model.rendering_network.grid_parameters()),
                "lr": self.lr * self.lr_factor_for_color_grid,
            },
            {"name": "net", "params": list(self.model.rendering_network.mlp_parameters()), "lr": self.lr},
            {"name": "density", "params": list(self.model.density.parameters()), "lr": self.learning_rate_beta},
            {
                "name": "coarse_mlp_parameters",
                "params": list(self.model.implicit_network.coarse.mlp_parameters()),
                "lr": self.lr,
            },
        ]
        self.optimizer = torch.optim.Adam(para_list, betas=(0.9, 0.99), eps=1e-15)

        # continue from previous checkpoint
        self.start_frame_idx = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, "checkpoints")

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, "ModelParameters", str(kwargs["checkpoint"]) + ".pth")
            )

            if "rendering_network.embeddings" in saved_model_state["model_state_dict"].keys():
                saved_model_state["model_state_dict"]["rendering_network.embeddings"] = saved_model_state[
                    "model_state_dict"
                ]["rendering_network.embeddings"][: self.n_images]
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            if "voxels" in saved_model_state.keys():
                self.model.voxels = saved_model_state["voxels"]
            if self.model.density_method == "volsdf_gridpredefined":
                self.model.density.voxels = self.model.voxels
                self.model.density.voxel_res = self.model.voxel_res

            self.start_frame_idx = saved_model_state["frame_idx"]
            print("Resuming from frame_idx: {0}".format(self.start_frame_idx))
            data = torch.load(
                os.path.join(old_checkpnts_dir, "OptimizerParameters", str(kwargs["checkpoint"]) + ".pth")
            )
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            pose_data = torch.load(
                os.path.join(old_checkpnts_dir, "PoseParameters", str(kwargs["checkpoint"]) + ".pth")
            )
            self.train_dataset.est_pose_all = pose_data["est_pose_all"]

        self.tracking_num_pixels = self.conf.get_int("train.tracking_num_pixels", default=1024)
        self.mapping_num_pixels = self.conf.get_int("train.mapping_num_pixels", default=10240)
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.plot_freq = self.conf.get_int("train.plot_freq")
        self.checkpoint_freq = self.conf.get_int("train.checkpoint_freq", default=100)
        self.split_n_pixels = self.conf.get_int("train.split_n_pixels", default=10000)
        self.plot_conf = self.conf.get_config("plot")
        self.enable_BA = self.conf.get_bool("SLAM.mapping.BA")
        self.BA_ratio = self.conf.get_float("SLAM.mapping.BA_ratio")
        self.BA_cam_lr = self.conf.get_float("SLAM.mapping.BA_cam_lr")
        self.cam_lr = self.conf.get_float("SLAM.tracking.lr")
        self.num_cam_iters = self.conf.get_int("SLAM.tracking.iters")
        self.num_mapping_iters = self.conf.get_int("SLAM.mapping.iters")
        self.model.num_mapping_iters = self.num_mapping_iters
        self.mapping_every_frame = self.conf.get_int("SLAM.mapping.mapping_every_frame")
        self.train_dataset.mode = ""

    def save_checkpoints(self, frame_idx):
        model_dict = {"frame_idx": frame_idx, "model_state_dict": self.model.state_dict(), "voxels": self.model.voxels}

        # uncomment to also save/keep intermediate checkpoints
        # torch.save(
        #     model_dict,
        #     os.path.join(self.checkpoints_path, self.model_params_subdir, str(frame_idx) + ".pth"))
        torch.save(model_dict, os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        # torch.save(
        #     {"frame_idx": frame_idx, "optimizer_state_dict": self.optimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(frame_idx) + ".pth"))
        torch.save(
            {"frame_idx": frame_idx, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"),
        )

        # torch.save(
        #     {"frame_idx": frame_idx, "est_pose_all": self.train_dataset.est_pose_all, "gt_pose_all": self.train_dataset.gt_pose_all},
        #     os.path.join(self.checkpoints_path, self.pose_params_subdir, str(frame_idx) + ".pth"))
        torch.save(
            {
                "frame_idx": frame_idx,
                "est_pose_all": self.train_dataset.est_pose_all,
                "gt_pose_all": self.train_dataset.gt_pose_all,
            },
            os.path.join(self.checkpoints_path, self.pose_params_subdir, "latest.pth"),
        )

    def vis(self, frame_idx, mode, inner_iter, c2w=None):
        """Visualize the model outputs. Rendering and meshing are done here."""

        plots_dir = self.plots_dir
        torch.cuda.empty_cache()
        self.model.eval()
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = self.train_dataset.collate_fn(
            [self.train_dataset[frame_idx % self.n_images]]
        )

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["pose"] = model_input["pose"].cuda() if c2w is None else c2w.unsqueeze(0).cuda()
        model_input["gt_depth"] = ground_truth["gt_depth"].cuda()
        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
        res = []
        for s in split[:]:

            out = self.model(s, indices, ground_truth, mode=mode + "_vis")

            d = {
                "rgb_values": out["rgb_values"].detach(),
                "normal_map": out["normal_map"].detach(),
                "depth_values": out["depth_values"].detach(),
            }
            if "rgb_un_values" in out:
                d["rgb_un_values"] = out["rgb_un_values"].detach()
            res.append(d)

        batch_size = ground_truth["rgb"].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        plot_data = self.get_plot_data(
            model_input,
            model_outputs,
            model_input["pose"],
            ground_truth["rgb"],
            ground_truth["normal"],
            ground_truth["depth"],
            ground_truth["gt_depth"],
        )

        plt.plot(
            self.model.implicit_network,
            self.model.rendering_network,
            indices,
            plot_data,
            plots_dir,
            frame_idx,
            self.img_res,
            inner_iter=inner_iter,
            save_mesh=(mode == "mapping"),
            **self.plot_conf,
        )

        self.model.train()

    def build_graph(self, keyframe_list, placeholder=0, thresh=30):
        """Build the graph for flow."""
        ides = []
        es = []
        for idx, x in enumerate(keyframe_list):
            for idy, y in enumerate(keyframe_list):
                if x % 10 == 0 and y % 10 == 0:
                    if 0 < abs(x - y) <= thresh:
                        ides.append((idx, idy))
                        es.append((x, y))
        idii, idjj = torch.as_tensor(ides, device="cuda:0").unbind(dim=-1)
        ii, jj = torch.as_tensor(es, device="cuda:0").unbind(dim=-1)
        return idii + placeholder, idjj + placeholder, ii, jj

    def get_edges_flow(self, edges):
        """Get the corresponding GT flow images with respect to the graph(edges) information."""
        idii, idjj, ii, jj = edges
        flows = []
        flow_masks = []
        for i, j in zip(ii, jj):
            path = f"{self.flow_dir}/{i:04d}_{j:04d}_flow.npy"
            try:
                with lzma.open(path, "rb") as f:
                    flow = torch.from_numpy(np.load(f)).float()
            except:
                flow = torch.from_numpy(np.load(path)).float()
            path = f"{self.flow_dir}/{i:04d}_{j:04d}_occ.png"
            flow_mask = cv2.imread(path)
            flow_mask = flow_mask[:, :, 0] == 0
            flow_mask = torch.from_numpy(flow_mask).bool()
            flows.append(flow)
            flow_masks.append(flow_mask)
        flows = torch.stack(flows)
        flow_masks = torch.stack(flow_masks)
        return flows, flow_masks

    def select_flow_uv(self, gt_edges_flow, gt_edges_flow_mask, model_input, edges):
        """Get the corresponding flow value of the selected pixels."""
        idii, idjj, ii, jj = edges
        sampling_idx = model_input["sampling_idx"]
        sampling_idx = sampling_idx[idii]
        bs = sampling_idx.shape[0]
        gt_edges_flow = gt_edges_flow.reshape(bs, -1, 2)
        gt_edges_flow_mask = gt_edges_flow_mask.reshape(bs, -1)
        idx1 = torch.arange(bs)
        idx1 = idx1.repeat_interleave(sampling_idx.shape[1])
        idx2 = sampling_idx.reshape(-1)
        gt_edges_flow = gt_edges_flow[idx1, idx2]
        gt_edges_flow_mask = gt_edges_flow_mask[idx1, idx2]
        return gt_edges_flow.reshape(bs, -1, 2), gt_edges_flow_mask.reshape(bs, -1)

    def run(self):
        gt_cam = self.conf.get_float("SLAM.tracking.gt_cam", default=False)
        print("running...")

        for frame_idx in tqdm(range(self.start_frame_idx, self.train_dataset.n_images)):
            self.train_dataset.frame_idx = frame_idx

            if frame_idx % self.checkpoint_freq == 0 and frame_idx != 0:
                self.save_checkpoints(frame_idx)

            # tracking
            self.train_dataset.mode = "tracking"
            gt_c2w = self.train_dataset.gt_pose_all[frame_idx]
            if (frame_idx == 0) or gt_cam:
                self.train_dataset.est_pose_all[frame_idx] = gt_c2w
            else:
                device = "cuda:0"
                if self.const_speed_assumption and frame_idx - 2 >= 0:
                    delta = (
                        self.train_dataset.est_pose_all[frame_idx - 1]
                        @ self.train_dataset.est_pose_all[frame_idx - 2].inverse()
                    )
                    estimated_new_cam_c2w = delta @ self.train_dataset.est_pose_all[frame_idx - 1]
                else:
                    estimated_new_cam_c2w = self.train_dataset.est_pose_all[frame_idx - 1]

                self.train_dataset.est_pose_all[frame_idx] = estimated_new_cam_c2w

                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())

                camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                cam_para_list = [camera_tensor]
                optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)

                scheduler_camera = StepLR(optimizer_camera, step_size=50, gamma=0.95)
                init_error_trans = torch.dist(gt_camera_tensor.to(device)[-3:], camera_tensor[-3:]).item()
                init_error_rot = get_error_degrees(gt_camera_tensor.to(device)[:-3], camera_tensor[:-3])

                candidate_cam_tensor = None
                current_min_loss = 10000000000.0
                if not self.tracking_change_pix_within_iters:
                    self.train_dataset.change_sampling_idx(self.tracking_num_pixels)
                for cam_iter in range(self.num_cam_iters):
                    c2w = get_camera_from_tensor(camera_tensor)

                    if self.tracking_change_pix_within_iters:
                        self.train_dataset.change_sampling_idx(self.tracking_num_pixels)

                    indices, model_input, ground_truth = self.train_dataset.collate_fn([self.train_dataset[frame_idx]])

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input["pose"] = c2w.unsqueeze(0)
                    model_outputs = self.model(model_input, indices, ground_truth, mode="tracking", frame_idx=frame_idx)
                    loss_output = self.tracking_loss(
                        model_outputs, ground_truth, stage="fine", frame_idx=frame_idx
                    )

                    loss = loss_output["loss"]
                    loss.backward()
                    optimizer_camera.step()
                    scheduler_camera.step()
                    optimizer_camera.zero_grad()

                    if cam_iter == 0:
                        initial_loss = loss

                    if self.verbose:
                        if (cam_iter == self.num_cam_iters - 1) or (cam_iter % 1 == 0):
                            error_trans = torch.dist(gt_camera_tensor.to(device)[-3:], camera_tensor[-3:]).item()
                            error_rot = get_error_degrees(gt_camera_tensor.to(device)[:-3], camera_tensor[:-3])
                            print(
                                f"Re-rendering loss: {float(initial_loss):.4f}->{float(loss):.4f} "
                                + f"camera error R: {init_error_rot:.4f}->{error_rot:.4f}"
                                + f" T: {init_error_trans:.4f}->{error_trans:.4f}"
                            )

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()

                c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
                self.train_dataset.est_pose_all[frame_idx] = c2w.cpu()

            # mapping
            if frame_idx % self.mapping_every_frame == 0:
                self.train_dataset.mode = "mapping"
                for mapping_iter in range(self.num_mapping_iters):
                    if frame_idx == 0:
                        self.BA = False
                    else:
                        self.BA = self.enable_BA and (mapping_iter > int(self.num_mapping_iters * self.BA_ratio))

                    # prepare keyframe list and flow edges
                    if frame_idx == 0:
                        keyframe_list = [0]
                        edges = None
                    else:
                        # global + local
                        if frame_idx < 200:
                            keyframe_list = []
                        else:
                            keyframe_list = (
                                torch.randint(
                                    max(frame_idx // self.keyframe_every - 4, 0), (self.mapping_window_size // 3,)
                                )
                                * self.keyframe_every
                            ).tolist()
                            keyframe_list = sorted(keyframe_list)

                        if mapping_iter == 0:
                            # local_list will always be the same across iterations
                            if frame_idx < 200:
                                local_list = list(range(0, frame_idx, 10)) + [frame_idx]
                            else:
                                local_list = (
                                    torch.randint(
                                        max(frame_idx // self.keyframe_every - 20, 0),
                                        frame_idx // self.keyframe_every,
                                        (self.mapping_window_size // 3 * 2,),
                                    )
                                    * self.keyframe_every
                                ).tolist()
                                local_list += [frame_idx]
                            local_list = sorted(list(set(local_list)))
                            if len(keyframe_list) >= 2:
                                edges = self.build_graph(local_list, placeholder=self.mapping_window_size // 3)
                                gt_edges_flow, gt_edges_flow_mask = self.get_edges_flow(edges)
                            else:
                                edges = None

                        # add normal frame to the BA, at later stage
                        if mapping_iter == self.num_mapping_iters // 2:
                            local_list += list(range(frame_idx // self.keyframe_every * self.keyframe_every, frame_idx))

                        keyframe_list += local_list
                    if self.verbose:
                        print("keyframe_list: ", keyframe_list)

                    # prep keyframe_list data
                    datas = []
                    camera_tensors = []
                    gt_camera_tensors = []
                    self.train_dataset.change_sampling_idx(self.mapping_num_pixels // len(keyframe_list))
                    for keyframe in keyframe_list:
                        data = self.train_dataset[keyframe]
                        datas.append(data)
                        if self.BA:
                            if keyframe == 0:
                                camera_tensor = get_tensor_from_camera(self.train_dataset.gt_pose_all[keyframe])
                            else:
                                camera_tensor = get_tensor_from_camera(self.train_dataset.est_pose_all[keyframe])
                            camera_tensors.append(camera_tensor)
                            gt_camera_tensor = get_tensor_from_camera(self.train_dataset.gt_pose_all[keyframe])
                            gt_camera_tensors.append(gt_camera_tensor)
                    indices, model_input, ground_truth = self.train_dataset.collate_fn(datas)

                    # if BA, add the camera tensors to the optimizer
                    if self.BA:
                        gt_camera_tensors = torch.stack(gt_camera_tensors).cuda()
                        camera_tensors = torch.stack(camera_tensors)
                        camera_tensors = Variable(camera_tensors.cuda(), requires_grad=True)
                        cam_para_list = [camera_tensors]
                        optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.BA_cam_lr)
                        model_input["pose"] = get_camera_from_tensor(camera_tensors)

                    # do visualization
                    if (
                        (frame_idx > 1)
                        and (mapping_iter % self.mapping_inner_freq == 0)
                        and (frame_idx % self.plot_freq == 0)
                    ):
                        self.vis(frame_idx, "mapping", mapping_iter)

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input["pose"] = model_input["pose"].cuda()
                    if edges is not None:
                        ground_truth["edges"] = edges
                        ground_truth["flow"], ground_truth["flow_mask"] = self.select_flow_uv(
                            gt_edges_flow, gt_edges_flow_mask, model_input, edges
                        )

                    self.optimizer.zero_grad()
                    if self.BA:
                        optimizer_camera.zero_grad()
                    if frame_idx > 1:
                        stage = "coarse" if mapping_iter < int(self.num_mapping_iters * 0.25) else "fine"
                        color_stage = "base" if mapping_iter < int(self.num_mapping_iters * 0.7) else "highfreq"
                    else:
                        stage = "fine"
                        color_stage = "highfreq"
                    model_outputs = self.model(
                        model_input,
                        indices,
                        ground_truth,
                        keyframe_list=keyframe_list,
                        frame_idx=frame_idx,
                        mode="mapping",
                        stage=stage,
                        color_stage=color_stage,
                        iter=mapping_iter,
                    )
                    loss_output = self.loss(
                        model_outputs,
                        ground_truth,
                        keyframe_list,
                        frame_idx=frame_idx,
                        stage=stage,
                    )
                    loss = loss_output["loss"]
                    loss.backward()
                    self.optimizer.step()
                    if self.BA:
                        optimizer_camera.step()
                    if self.BA:
                        if self.verbose:
                            error = torch.mean(torch.abs(gt_camera_tensors - camera_tensors.detach()), -1)
                            print("Error:", error, "Mean error:", error.mean())
                        # after BA put the poses back to the est_pose_all list
                        poses = get_camera_from_tensor(camera_tensors).detach().cpu()
                        for ii, keyframe in enumerate(keyframe_list):
                            if keyframe == 0:
                                # frames to always give GT pose
                                self.train_dataset.est_pose_all[keyframe] = self.train_dataset.gt_pose_all[keyframe]
                            elif not (
                                frame_idx >= 1 and (keyframe in keyframe_list[: self.mapping_window_size // 2])
                            ):
                                self.train_dataset.est_pose_all[keyframe] = poses[ii]

                    if self.verbose:
                        psnr = rend_util.get_psnr(model_outputs["rgb_values"], ground_truth["rgb"].cuda())

                        print(
                            "{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, gt_depth_loss = {9}, warp_loss = {10}, smooth_loss = {11}, flow_loss = {12}, psnr = {8}".format(
                                self.expname,
                                self.timestamp,
                                frame_idx,
                                mapping_iter,
                                self.num_mapping_iters,
                                loss.item(),
                                loss_output["rgb_loss"].item(),
                                float(loss_output["eikonal_loss"]),
                                psnr.item(),
                                loss_output["gt_depth_loss"],
                                loss_output["warp_loss"],
                                float(loss_output["smooth_loss"]),
                                loss_output["flow_loss"],
                            )
                        )

            if frame_idx % self.mapping_every_frame != 0:
                self.train_dataset.clean(frame_idx)

        self.save_checkpoints(frame_idx)
        self.vis(frame_idx, "mapping", 0)

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, depth_real_gt):
        """Prepare data for plotting."""
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs["rgb_values"].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs["normal_map"].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.0) / 2.0

        depth_map = model_outputs["depth_values"].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_gt, depth_map[..., None], depth_gt > 0.0)
        depth_gt = depth_gt * scale + shift

        depth_real_gt = depth_real_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_real_gt, depth_map[..., None], depth_real_gt > 0.0)
        depth_real_gt = depth_real_gt * scale + shift

        plot_data = {
            "rgb_gt": rgb_gt,
            "normal_gt": (normal_gt + 1.0) / 2.0,
            "depth_gt": depth_gt,
            "depth_real_gt": depth_real_gt,
            "pose": pose,
            "rgb_eval": rgb_eval,
            "normal_map": normal_map,
            "depth_map": depth_map,
        }

        if "rgb_un_values" in model_outputs:
            plot_data["rgb_un_eval"] = model_outputs["rgb_un_values"].reshape(batch_size, num_samples, 3)

        return plot_data
