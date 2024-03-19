<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="https://nicer-slam.github.io/static/images/72.png" width="60">NICER-SLAM: Neural Implicit Scene Encoding for RGB SLAM</h1>
  <p align="center">
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    ·
    <a href="https://pengsongyou.github.io"><strong>Songyou Peng*</strong></a>
    ·
    <a href="http://people.inf.ethz.ch/vlarsson/"><strong>Viktor Larsson</strong></a>
    ·
    <a href="https://zhpcui.github.io/"><strong>Zhaopeng Cui</strong></a>
    ·
    <a href="http://people.inf.ethz.ch/moswald/"><strong>Martin R. Oswald</strong></a>
</p>
<p align="center">
    <a href="https://www.cvlibs.net/"><strong>Andreas Geiger</strong></a>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
</p>

  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h2 align="center">3DV 2024 (Oral)</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2302.03594">Paper</a> | <a href="https://youtu.be/H4cOCa3oUno">Video</a> | <a href="https://nicer-slam.github.io">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./media/nicer-slam auto slide teaser.gif" alt="Logo" width="80%">
  </a>
</p>

<p align="center">
NICER-SLAM produces accurate dense geometry and camera tracking without the need of depth sensor input.
</p>
<p align="center">
(The black / red lines are the ground truth / predicted camera trajectory)
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#visualizing-nicer-slam-results">Visualization</a>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `nicer-slam`.
```bash
conda env create -f env_yamls/nicer-slam.yaml
conda activate nicer-slam
```

## Visualizing NICER-SLAM Results
We provide the results of NICER-SLAM ready for download. You can run our **interactive visualizer** as following. 

### Self-captured Outdoor Dataset
To visualize our results on the Self-captured Outdoor Dataset:
```bash
bash scripts/download_vis_sco.sh
# Choose one of the following scenes
OUTPUT_FOLDER=exps/azure_2/2024_03_11_22_57_38sample_vis
OUTPUT_FOLDER=exps/azure_3/2024_03_11_22_57_38sample_vis
python visualizer.py --output $OUTPUT_FOLDER
```
<p align="center">
  <img src="./media/azure.gif" width="60%" />
</p>

<p align="center">
  <img src="./media/azure2.gif" width="60%" />
</p>

You can find more results of NICER-SLAM [here](https://cvg-data.inf.ethz.ch/nicer-slam/vis).

### 7-Scenes
```bash
bash scripts/download_vis_7scenes_office.sh
OUTPUT_FOLDER=exps/7scenes_4/2024_02_27_15_05_29sample_vis
python visualizer.py --output $OUTPUT_FOLDER
```
<p align="center">
  <img src="./media/7scenes.gif" width="60%" />
</p>

### Replica
```bash
bash scripts/download_vis_replica_room2.sh
OUTPUT_FOLDER=exps/replica_3/2024_02_27_00_12_45sample_vis
python visualizer.py --output $OUTPUT_FOLDER
```
<p align="center">
  <img src="./media/replica.gif" width="70%" />
</p>

### Interactive Visualizer Usage
The black trajectory indicates the ground truth trajectory, and the red is trajectory of NICER-SLAM.
- Press `Ctrl+0` for grey mesh rendering.
- Press `Ctrl+1` for textured mesh rendering.
- Press `Ctrl+9` for normal rendering.
- Press `L` to turn off/on lighting.
### Command Line Arguments
- `--output $OUTPUT_FOLDER` output folder
- `--save_rendering` save rendering video to `vis.mp4` in the output folder
- `--render_every_frame` screen recording speed and frame rate are syncd, one frame in video corresponds to one frame in input
- `--no_gt_traj` do not show ground truth trajectory

## Demo

Here you can run NICER-SLAM yourself on two short sequences with 200 frames. 

First, download the demo data as below and the data is saved into the `./Datasets/processed/Demo` folder.
```bash
bash scripts/download_demo.sh
```
Next, run NICER-SLAM. It takes about half an hour with ~24G GPU memory.

For short sequence from Self-captured Outdoor Dataset.
```bash
cd code
python training/exp_runner.py --conf confs/runconf_demo_1.conf
```
For short sequence from Replica Dataset.
```bash
cd code
python training/exp_runner.py --conf confs/runconf_demo_2.conf
```
Finally, run the following command to visualize.
```bash 
# here is just an example, change to your output folder
OUTPUT_FOLDER=exps/demo_1/2024_03_16_16_39_08demo
python code/evaluation/eval_cam.py --output $OUTPUT_FOLDER
python visualizer.py --output $OUTPUT_FOLDER
```

**NOTE:** This is for demonstration only, its configuration/performance may be different from our paper.


## Third Party 

Install third-party repositories for monocular depth/normal estimation and optical flow extraction.

```bash
mkdir 3rdparty
cd 3rdparty
git clone https://github.com/EPFL-VILAB/omnidata.git
cd omnidata/omnidata_tools/torch
mkdir -p pretrained_models && cd pretrained_models
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt'
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt'
cd ..
cd ../../../
git clone https://github.com/haofeixu/gmflow.git
cd gmflow
```

Download GMFlow's pretrained models from https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view?usp=sharing to the `gmflow` folder. Then unzip it.
```bash
unzip pretrained.zip
cd ../../
```

Install the environments for the third-party repositories.
```bash
conda env create -f env_yamls/gmflow.yaml
conda env create -f env_yamls/omnidata.yaml
```


## Run

### Replica Dataset
Download the data as below and the data is saved into the `./Datasets/orig/Replica` folder. Note that the Replica data is generated by the authors of iMAP, so please cite iMAP if you use the data.
```bash
bash scripts/download_replica.sh
```
Run the following command to preprocess the data. This includes converting the camera pose to VolSDF format, extracting monocular depth/normal estimation and optical flow. The processed data is saved into the `./Datasets/processed/Replica` folder. 
NOTE that the npy file for depth/normal/flow is compressed with [lzma](https://docs.python.org/3/library/lzma.html) to save storage.
```bash
python preprocess/replica_2_volsdf.py
```
Then you can run NICER-SLAM:
```bash
cd code
python training/exp_runner.py --conf confs/replica/runconf_replica_2.conf
```

### 7-Scenes Dataset
Download the data as below and the data is saved into the `./Datasets/orig/7Scenes` folder. 
```bash
bash scripts/download_7scenes.sh
```
The 7-Scenes dataset does not provide meshes. We provide the following code to do TSDF-Fusion on the gt pose and depth image to get the mesh.
```bash
python preprocess/get_mesh_7scenes.py
```
Run the following command to preprocess the data. This includes converting the camera pose to VolSDF format, extracting monocular depth/normal estimation and optical flow. The processed data is saved into the `./Datasets/processed/7Scenes` folder. 
```bash
python preprocess/7scenes_2_volsdf.py
```
Then you can run NICER-SLAM:
```bash
cd code
python training/exp_runner.py --conf confs/7scenes/runconf_7scenes_2.conf
```

### Self-captured Outdoor Dataset
Download the data as below and the data is saved into the `./Datasets/orig/Azure` folder. 
```bash
bash scripts/download_azure.sh
```
Run the following command to preprocess the data. This includes converting the camera pose to VolSDF format, extracting monocular depth/normal estimation and optical flow. The processed data is saved into the `./Datasets/processed/Azure` folder. 
```bash
python preprocess/azure_2_volsdf.py
```
Then you can run NICER-SLAM:
```bash
cd code
python training/exp_runner.py --conf confs/azure/runconf_azure_2.conf
```
**NOTE:** Please ensure your GPU has over 30GB of memory to avoid surpassing the GPU memory limit. Reducing the batch size can help lessen memory consumption, though it will require more iterations to achieve comparable performance.

## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding output folder name:
```bash
# assign any output_folder you like, here is just an example
OUTPUT_FOLDER=exps/replica_4/2024_03_01_00_07_16code_release
python code/evaluation/eval_cam.py --output $OUTPUT_FOLDER
```
**NOTE:** For Self-captured Outdoor Dataset, the error is in COLMAP's metric system, not in `cm`.

### Reconstruction Error
Unlike NICE-SLAM that evaluates on culled mesh, in NICER-SLAM we directly evaluates on the original mesh to showcase the extrapolation ability of our reconstruction.
```bash
# assign any output_folder you like, here is just an example
OUTPUT_FOLDER=exps/replica_4/2024_03_01_00_07_16code_release
python code/evaluation/eval_rec.py --output $OUTPUT_FOLDER
```

### Rendering Evaluation

#### Interpolation
Evaluate the rendering on non-keyframes of input sequences, in default it selects the indexs `range(2, n_imgs, 100)`.  
```bash
# assign any run, here is just an example
TIMESTAMP=2024_03_01_00_07_16code_release
CONF=../exps/replica_4/2024_03_01_00_07_16code_release/runconf.conf
cd code
python evaluation/eval_rendering.py --conf $CONF --checkpoint latest --is_continue --timestamp $TIMESTAMP --eval_method interpolate
```
#### Extrapolation
To evaluate the method's ability to rendering on extrapolated views. We made an extrapolation rendering evaluation set on the Replica dataset using the Replica Renderer. It should be downloaded while downloading the replica dataset and saved into the folder `Datasets/orig/Replica_eval_ext`. Run the following command to preprocess it to VolSDF coordinate system.
```bash
python preprocess/replica_eval_2_volsdf.py
```
Now we are ready to evaluate.
```bash
# assign any run, here is just an example
TIMESTAMP=2024_03_01_00_07_16code_release
CONF=../exps/replica_4/2024_03_01_00_07_16code_release/runconf.conf
cd code
python evaluation/eval_rendering.py --conf $CONF --checkpoint latest --is_continue --timestamp $TIMESTAMP --eval_method extrapolate
```

## Acknowledgement
We adapted some codes from some awesome repositories including [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), [lietorch](https://github.com/princeton-vl/lietorch), [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF), [VolSDF](https://github.com/lioryariv/volsdf), [MonoSDF](https://github.com/autonomousvision/monosdf), [MiDaS](https://github.com/isl-org/MiDaS) and [DIST-Renderer](https://github.com/B1ueber2y/DIST-Renderer). Thanks for making codes public available. 

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{Zhu2024NICER,
  author={Zhu, Zihan and Peng, Songyou and Larsson, Viktor and Cui, Zhaopeng and Oswald, Martin R and Geiger, Andreas and Pollefeys, Marc},
  title     = {NICER-SLAM: Neural Implicit Scene Encoding for RGB SLAM},
  booktitle = {International Conference on 3D Vision (3DV)},
  month     = {March},
  year      = {2024},
}
```
## Contact
Contact [Zihan Zhu](mailto:zhuzihan2000@gmail.com) and [Songyou Peng](mailto:songyou.pp@gmail.com) for questions, comments and reporting bugs.
