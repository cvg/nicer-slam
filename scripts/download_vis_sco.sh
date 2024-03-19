#!/bin/bash

mkdir -p exps
cd exps
wget https://cvg-data.inf.ethz.ch/nicer-slam/vis/azure_2.zip
wget https://cvg-data.inf.ethz.ch/nicer-slam/vis/azure_3.zip
unzip azure_2.zip
unzip azure_3.zip
rm azure_2.zip
rm azure_3.zip