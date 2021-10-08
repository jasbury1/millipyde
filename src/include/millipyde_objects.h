#ifndef MILLIPYDE_OBJECTS_H
#define MILLIPYDE_OBJECTS_H

#include "millipyde.h"

#define CHECK_GPU_OBJECT(o) (PyObject_TypeCheck(o, &PyGPUArray_Type) || PyObject_TypeCheck(o, &PyGPUImage_Type))

#ifdef __cplusplus
extern "C"
{
#endif

    //PyObject * gpuarray_transpose(PyObject *array);
    void 
    mpobj_copy_from_host(MPObjData *obj_data, void *data, size_t nbytes);

    void *
    mpobj_copy_to_host(MPObjData *obj_data);

    void 
    mpobj_change_device(MPObjData *obj_data, int device_id);

    void 
    mpobj_dealloc_device_data(MPObjData *obj_data);


#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_OBJECTS_H