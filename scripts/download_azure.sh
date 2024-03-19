#!/bin/bash

mkdir -p Datasets/orig
cd Datasets/orig
wget https://cvg-data.inf.ethz.ch/nicer-slam/data/Azure.zip
unzip Azure.zip
rm -rf Azure.zip