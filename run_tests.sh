#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

rebuild=false
run_python=true
run_backend=true
multi=false

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Runs the millipyde test suite"
      echo " "
      echo "options:"
      echo "-h, --help                this screen right here"
      echo "-m, --multigpu            include tests for 2+ GPUs"
      echo "-b, --build               rebuild before running"
      echo "-p, --python              run only the python tests"
      echo "-c, --c                   run only the c/c++ tests"
      exit 0
      ;;
    -b|--build)
      rebuild=true
      shift
      ;;
    -m|--multigpu)
       multi=true
       shift
       ;;
    -p|--python)
      run_backend=false
      shift
      ;;
    -c|--c)
      run_python=false
      shift
      ;;
    *)
      break
      ;;
  esac
done

if  [ "$run_python" = true ] ; then
  if  [ "$rebuild" = true ] ; then
    ./build.sh
  fi
  echo -e "\n${CYAN}Running Python Unittests...${NC}\n"
  python3 tests/millipyde_tests.py -v
  if  [ "$multi" = true ] ; then
    echo -e "\n${CYAN}Running Python Unittests for multi-GPU systems...${NC}\n"
    python3 tests/millipyde_multigpu_tests.py -v
  fi
fi

#if  [ "$run_backend" = true ] ; then
#  if  [ "$rebuild" = true ] ; then
#    cd ./tests/backend/
#    make clean
#    make test
#    cd -
#  fi
#  echo -e "\n${CYAN}Running Backend Unittests...${NC}\n"
#fi




      

