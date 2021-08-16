#Dependencies : [MYHIP]/bin must be in user's path.

HIP_PATH?= $(wildcard /opt/rocm-4.1.0/hip)
SOURCES_PATH=//home/jasbury/Thesis/millipyde/src

ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

CCFLAGS = -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv \
-O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security \
-g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security \
-Wdate-time -D_FORTIFY_SOURCE=2 -fPIC

LINKFLAGS = -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,\
-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 -Wl,-Bsymbolic-functions \
-Wl,-z,relro -g -fwrapv -O2 -g -fstack-protector-strong -Wformat \
-Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2

SRC      := $(wildcard $(SOURCES_PATH)/*.cpp)
INCLUDE  := -Iinclude/ \
-I/usr/local/lib/python3.8/dist-packages/numpy/core/include \
-I$(SOURCES_PATH)/include/ \
-I/opt/rocm-4.1.0/hip/include/hip \
-I/usr/include/python3.8
#LINK := -L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu

# Show how to use PLATFORM to specify different options for each compiler:
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20
endif

EXE=test
objects = gpuarray.o millipyde_image.o gpuarray_funcs.o bit_extract.o test_cases.o

gpuarray.o: $(SOURCES_PATH)/gpuarray.c
	$(HIPCC) $(HIPCC_FLAGS) $(CCFLAGS) $(INCLUDE) -c $< -o $@

millipyde_image.o: $(SOURCES_PATH)/millipyde_image.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CCFLAGS) $(INCLUDE) -c $< -o $@

gpuarray_funcs.o: $(SOURCES_PATH)/gpuarray_funcs.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CCFLAGS) $(INCLUDE) -c $< -o $@

bit_extract.o: $(SOURCES_PATH)/bit_extract.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CCFLAGS) $(INCLUDE) -c $< -o $@

test_cases.o: test_cases.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CCFLAGS) $(INCLUDE) -c $< -o $@

test: $(objects)
	$(HIPCC) $(HIPCC_FLAGS) $(LINKFLAGS) $(objects) -o $(EXE)

clean:
	rm -f *.o $(EXE)
