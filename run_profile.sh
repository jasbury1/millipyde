#!/bin/bash
rocprof --timestamp on --basenames on --stats --hip-trace --sys-trace -o profile/results.csv python3 examples/image_examples.py
