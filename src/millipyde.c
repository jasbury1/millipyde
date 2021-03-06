#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>
#include <limits.h>

#include "millipyde.h"

const char *mperr_str(MPStatus status)
{
    switch (status)
    {
    case MOD_ERROR:
        return "Could not import module 'millipyde' due to module creation failure.";
    case MOD_ERROR_CREATE_GPUARRAY_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'gpuarray'";
    case MOD_ERROR_CREATE_GPUIMAGE_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'gpuimage'";
    case MOD_ERROR_CREATE_OPERATION_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'Operation'";
    case MOD_ERROR_CREATE_PIPELINE_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'Pipeline'";
    case MOD_ERROR_CREATE_DEVICE_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'Device'";
    case MOD_ERROR_CREATE_GENERATOR_TYPE:
        return "Could not import module 'millipyde' while creating internal type 'Generator'";
    case MOD_ERROR_ADD_GPUARRAY:
        return "Could not import module 'millipyde' while loading internal type 'gpuarray'";
    case MOD_ERROR_ADD_GPUIMAGE:
        return "Could not import module 'millipyde' while loading internal type 'gpuimage'";
    case MOD_ERROR_ADD_OPERATION:
        return "Could not import module 'millipyde' while loading internal type 'Operation'";
    case MOD_ERROR_ADD_PIPELINE:
        return "Could not import module 'millipyde' while loading internal type 'Pipeline'";
    case MOD_ERROR_ADD_DEVICE:
        return "Could not import module 'millipyde' while loading internal type 'Device'";
    case MOD_ERROR_ADD_GENERATOR:
        return "Could not import module 'millipyde' while loading internal type 'Generator'"; 

    case DEV_ERROR_CURRENT_DEVICE:
        return "GPU runtime failed while querying the current device";
    case DEV_ERROR_DEVICE_COUNT:
        return "GPU runtime failed while querying the device count";
    case DEV_ERROR_DEVICE_PROPERTIES:
        return "GPU runtime failed while querying device properties";
    case DEV_ERROR_PEER_ACCESS_MATRIX_ALLOC:
        return "Could not allocate internal data-structure 'peer access matrix'";
    case DEV_ERROR_DEVICE_ARRAY_ALLOC:
        return "Coult not allocate internal data-structure 'device array'";
    case DEV_WARN_NO_PEER_ACCESS:
        return "Multiple devices were detected, but peer2peer is not supported on this system";
    case DEV_WARN_BAD_DEVICE:
        return "A device was detected but found to be unusable";

    case WORK_ERROR_INIT_PTHREAD:
        return "Unable to initialize work pool thread";
    case WORK_ERROR_INIT_COND:
        return "Unable to initialize work pool condition";
    case WORK_ERROR_INIT_MUX:
        return "Unable to initialize work pool mutex";
    case WORK_ERROR_ALLOC_WORK_POOL:
        return "Unable to allocate memory for work pool";
    case WORK_ERROR_ALLOC_WORK_NODE:
        return "Unable to allocate memory for worker node";
    case WORK_ERROR_NULL_WORK_POOL:
        return "The specified work pool was NULL";

    case TYPE_ERROR_NON_GPUOBJ:
        return "Cannot perform GPU operations on non-GPU-compatible type";

    case RAND_ERROR_INSUFFICIENT_BYTES:
        return "Random number generation failed to supply requrested bytes";

    case GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE:
        return "Constructing gpuarray requires an ndarray or array compatible argument";
    case GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_NUMERIC_ARRAY:
        return "Constructing gpuarray requires a numeric array type";

    case GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE:
        return "Construcing gpuimage requires an ndarray or array compatible argument";
    case GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_FORMAT:
        return "Construcing gpuimages requires a compatible image format";
    case GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_DIMS:
        return "Construcing gpuimages either a 2 dimensional array image "
               "format for single channel images (greyscale), or a 3 "
               "dimensional array image format for multi-channel images "
               "(rgb/rgba).";

    case GPUOPERATION_ERROR_CONSTRUCTION_NO_ARGS:
        return "Contructing Operations requires a runnable/callable argument";
    case GPUOPERATION_ERROR_INVALID_PROBABILITY:
        return "Constructing Operation requires a float probability between 0 and 1 (exclusive)";
    case GPUOPERATION_ERROR_CONSTRUCTION_NAMED_ARGS:
        return "Constructing Operations can only include one named argument designated 'probability'";
    case GPUOPERATION_ERROR_RUN_WITHOUT_STRING_METHOD:
        return "Operations must be constructed with a string method name to run as instance method";
    case GPUOPERATION_ERROR_RUN_UNKNOWN_STRING_METHOD:
        return "Operation's string method name could not be found for the given object";
    case GPUOPERATION_ERROR_RUN_NO_DEV_RANDOM:
        return "Unable to use /dev/random for random number generation";
    case GPUOPERATION_ERROR_RUN_CANNOT_READ_DEV_RANDOM:
        return "Unable to read from /dev/random for random number generation";


    case GPUPIPELINE_ERROR_CONSTRUCTION_INVALID_ARGS:
        return "Constructing Pipeline requires 2 arguments, or 3 arguments for specifying a device";
    case GPUPIPELINE_ERROR_INVALID_DEVICE:
        return "Constructing Pipeline requires an integer device";
    case GPUPIPELINE_ERROR_UNUSABLE_DEVICE:
        return "Constructing Pipeline requires a device that is useable for GPU operations";
    case GPUPIPELINE_ERROR_CONSTRUCTION_NAMED_ARGS:
        return "Constructing Pipelines can only include one named argument designated 'device'";
    case GPUPIPELINE_ERROR_NONLIST_INPUTS:
        return "Constructing Pipeline requires a List of inputs";
    case GPUPIPELINE_ERROR_NONLIST_OPERATIONS:
        return "Constructing Pipeline requires a List of operations";
    case GPUPIPELINE_ERROR_NONGPU_INPUT:
        return "Constructing Pipeline requires all inputs to be GPU compatible";

    case GPUGENERATOR_ERROR_INVALID_DEVICE:
        return "Constructing Generator requires an integer device";
    case GPUGENERATOR_ERROR_INVALID_MAX:
        return "Constructing Generator requires an integer number of outputs greater than 0";
    case GPUGENERATOR_ERROR_INVALID_RETURN_TO:
        return "Constructing Generator requries boolean value for whether to return_to_host";
    case GPUGENERATOR_ERROR_UNUSABLE_DEVICE:
        return "Constructing Generator requires a device that is useable for GPU operations";
    case GPUGENERATOR_ERROR_CONSTRUCTION_NAMED_ARGS:
        return "Constructing Generator can only include the named arguments 'device' and/or 'outputs'";
    case GPUGENERATOR_ERROR_INVALID_INPUT:
        return "Constructing Generator requires an list of inputs or a path to files";
    case GPUGENERATOR_ERROR_NONLIST_OPERATIONS:
        return "Constructing Generator requires a list of Operations";


    default:
        return "Unknown failure occurred";
    }
}

MPStatus
random_int_in_range(int min, int max, int *result)
{
    unsigned long buffer;
    ssize_t bytes;

    bytes = getrandom(&buffer, sizeof(unsigned long), 0);
    if (bytes < 0)
    {
        return RAND_ERROR_INSUFFICIENT_BYTES;
    }
    int rand = buffer % (max + 1 - min) + min;
    *result = rand;
    
    return MILLIPYDE_SUCCESS;
}

MPStatus  
random_double_in_range(double min, double max, double *result)
{
    unsigned long buffer;
    ssize_t bytes;

    bytes = getrandom(&buffer, sizeof(unsigned long), 0);
    if (bytes < 0)
    {
        return RAND_ERROR_INSUFFICIENT_BYTES;
    }

    double rand = (double)buffer / (double)ULONG_MAX;
    *result = (rand * (max - min)) + min;

    return MILLIPYDE_SUCCESS;
}