#!/bin/bash
/opt/rocm-4.1.0/bin/rocprof --hip-trace --sys-trace -o profile/results.csv python3 examples/image_examples.py
