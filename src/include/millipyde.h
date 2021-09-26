#ifndef MP_MILLIPYDE_H
#define MP_MILLIPYDE_H

#define MP_TRUE 1
#define MP_FALSE 0

// Macro representing for indicating the host as a location
#define HOST_LOC -1

typedef enum mp_status_codes {
    MILLIPYDE_SUCCESS = 0,

    MOD_ERROR,
    MOD_ERROR_CREATE_GPUARRAY_TYPE,
    MOD_ERROR_CREATE_GPUIMAGE_TYPE,
    MOD_ERROR_CREATE_OPERATION_TYPE,
    MOD_ERROR_CREATE_PIPELINE_TYPE,
    MOD_ERROR_ADD_GPUARRAY,
    MOD_ERROR_ADD_GPUIMAGE,
    MOD_ERROR_ADD_OPERATION,
    MOD_ERROR_ADD_PIPELINE,

    DEV_ERROR_CURRENT_DEVICE,
    DEV_ERROR_DEVICE_COUNT,
    DEV_ERROR_DEVICE_PROPERTIES,
    DEV_ERROR_PEER_ACCESS_MATRIX_ALLOC,
    DEV_WARN_NO_PEER_ACCESS,

    GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE,
    GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_NUMERIC_ARRAY,

    GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE,
    GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_FORMAT,
    GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_DIMS,

    GPUOPERATION_ERROR_CONSTRUCTION_NO_ARGS,
    GPUOPERATION_ERROR_INVALID_PROBABILITY,
    GPUOPERATION_ERROR_CONSTRUCTION_NAMED_ARGS,
    GPUOPERATION_ERROR_RUN_WITHOUT_STRING_METHOD,
    GPUOPERATION_ERROR_RUN_UNKNOWN_STRING_METHOD,
    GPUOPERATION_ERROR_RUN_NO_DEV_RANDOM,
    GPUOPERATION_ERROR_RUN_CANNOT_READ_DEV_RANDOM,

    GPUPIPELINE_ERROR_NONLIST_INPUTS,
    GPUPIPELINE_ERROR_NONLIST_OPERATIONS, 



} MPStatus;

typedef int MPBool;

const char* mperr_str(MPStatus status);

#endif // MP_MILLIPYDE_H