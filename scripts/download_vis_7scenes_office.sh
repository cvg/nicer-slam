#!/bin/bash

mkdir -p exps
cd exps
wget https://cvg-data.inf.ethz.ch/nicer-slam/vis/7scenes_4.zip
unzip '7scenes_4.zip'
rm '7scenes_4.zip'