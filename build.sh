#!/bin/bash
if [ ! -d "./build/" ] 
then
    # First time building
    echo "Creating build directory...\n"
fi

rm -rf ./lib/*.so
rm -rf ./build
python3 setup.py build_ext --inplace
