#!/bin/bash

# Define the base directory and create it
base_dir="Datasets/orig/7Scenes"
mkdir -p "${base_dir}"
cd "${base_dir}"

# Define an array of dataset names
datasets=(chess fire heads office pumpkin redkitchen stairs tsdf)

# Base URL for downloading the datasets
base_url="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

# Loop over the dataset array
for dataset in "${datasets[@]}"; do
    # Download the dataset zip file
    wget "${base_url}/${dataset}.zip"

    # Unzip the dataset
    unzip "${dataset}.zip"
    rm "${dataset}.zip"  # Remove the zip file after extracting

    # Special handling for datasets that contain a seq-01.zip
    if [ -d "${dataset}" ]; then
        cd "${dataset}"
        if [ -f "seq-01.zip" ]; then
            unzip "seq-01.zip"
            rm "seq-01.zip"  # Remove the seq-01.zip file after extracting
        fi
        cd ..
    fi
done