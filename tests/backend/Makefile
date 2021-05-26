#Dependencies : [MYHIP]/bin must be in user's path.

HIP_PATH?= $(wildcard /opt/rocm-4.1.0/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

SRC      := $(wildcard ../../src/*.cpp)
INCLUDE  := -Iinclude/

# Show how to use PLATFORM to specify different options for each compiler:
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20
endif

EXE=backend_millipyde_tests

$(EXE):
	$(HIPCC) -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.8/dist-packages/numpy/core/include -I../../src/include/ -I/opt/rocm-4.1.0/hip/include/hip -I/usr/include/python3.8 $(wildcard ../../src/*.cpp) $(HIPCC_FLAGS) $< -o $@

all: $(EXE)

clean:
	rm -f *.o $(EXE)