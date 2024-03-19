#!/bin/bash

mkdir -p Datasets/orig
cd Datasets/orig
# you can also download the Replica.zip manually through
# link: https://caiyun.139.com/m/i?1A5Ch5C3abNiL password: v3fY (the zip is split into smaller zips because of the size limitation of caiyun)
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm -rf Replica.zip
# download the pose/images for evaluation on extrapolated views
wget https://cvg-data.inf.ethz.ch/nicer-slam/data/Replica_eval_ext.zip
unzip Replica_eval_ext.zip
rm -rf Replica_eval_ext.zip