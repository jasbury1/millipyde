#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}Running Python Unittests...${NC}\n"

python3 tests/millipyde_tests.py -v

echo -e "\n${CYAN}Running Backend Unittests...${NC}\n"
