#!/bin/bash
rm -rf ./lib/*.so
rm -rf ./build
python setup.py build_ext --inplace