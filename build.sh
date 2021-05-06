#!/bin/bash
rm -rf ./lib/*.so
rm -rf ./build
python3 setup.py build_ext --inplace
