#!/bin/bash
rocprof -i profile/input.xml --timestamp on --basenames on --stats -o profile/results.csv python3 examples/image_examples.py
