#!/bin/bash

mkdir -p Datasets/processed
cd Datasets/processed
wget https://cvg-data.inf.ethz.ch/nicer-slam/data/Demo.zip
unzip Demo.zip
rm -rf Demo.zip