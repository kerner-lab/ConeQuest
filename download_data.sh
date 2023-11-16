#!/bin/bash

echo Downloaing data..!!

# Create data directory
mkdir data

# Download all data from Zenodo
wget -P data https://zenodo.org/api/records/10077410/files-archive

# Unzip the data
unzip data/files-archive -d data

# Use a for loop to iterate over each file in the directory
directory_path="data"

for file in "$directory_path"/*
do
    if [[ $file == *"zip"* ]]; then
        echo $file
        unzip -q $file -d $directory_path
        rm $file
    fi
done

rm data/files-archive
