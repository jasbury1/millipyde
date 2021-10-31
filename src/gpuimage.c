#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>
#include <dirent.h>
#include <string.h>
// #include <strings.h> ??

#include "gpuarray.h"
#include "gpuimage.h"
#include "millipyde_image.h"
#include "millipyde_devices.h"
#include "millipyde_objects.h"
#include "use_numpy.h"
#include "millipyde.h"

int
PyGPUImage_init(PyGPUImageObject *self, PyObject *args, PyObject *kwds) {
    PyObject *any = NULL;
    PyArrayObject *array = NULL;
    MPObjData *obj_data;

    if (!PyArg_ParseTuple(args, "O", &any))
    {
        return -1;
    }
    if (!any)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
        return -1;
    }
    // Typecast to PyArrayObject if it is already of that type
    if (PyArray_Check(any))
    {
        array = (PyArrayObject *)any;
    }
    // Attempt to create an array from the type passed to initializer
    else
    {
        array = (PyArrayObject *)PyArray_FROM_OTF(any, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
        if (array == NULL)
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
            return -1;
        }
    }
    if (!PyArray_ISNUMBER(array))
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_FORMAT));
        return -1;
    }

    int ndims = PyArray_NDIM(array);

    if (ndims != 2 && ndims != 3)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_DIMS));
        return -1;
    }

    if (PyGPUArray_Type.tp_init((PyObject *)self, args, kwds) < 0)
    {
        return -1;
    }

    obj_data = self->array.obj_data;
    self->width = obj_data->dims[1];
    self->height = obj_data->dims[0];
    return 0;
}

/*******************************************************************************
 * Calls the greyscale kernel on the gpuimage. If the image is not pinned to a
 * specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be greyscaled
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_color_to_greyscale(PyGPUImageObject *self, void *closure)
{
    // TODO: Type checking, etc
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // The greyscale function will set the current device based on obj_data
    mpimg_color_to_greyscale(obj_data, NULL);
    return Py_None;
}


/*******************************************************************************
 * Calls the transposition kernel on the gpuimage. If the image is not pinned to 
 * a specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_transpose(PyGPUImageObject *self, void *closure)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_transpose(obj_data, NULL);
    return Py_None;
}


/*******************************************************************************
 *
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_fliplr(PyGPUImageObject *self, void *closure)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_fliplr(obj_data, NULL);
    return Py_None;
}


/*******************************************************************************
 *
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_rotate(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    void *rotate_args = gpuimage_rotate_args(args);
    if (rotate_args == NULL)
    {
        return NULL;
    }

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_rotate(obj_data, rotate_args);

    free(rotate_args);
    return Py_None;
}


/*******************************************************************************
 * Calls the gaussian blur kernel on the gpuimage. If the image is not pinned to 
 * a specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_gaussian(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    void *gaussian_args = gpuimage_gaussian_args(args);
    if (gaussian_args == NULL)
    {
        return NULL;
    }

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_gaussian(obj_data, gaussian_args);

    free(gaussian_args);
    return Py_None;
}


/*******************************************************************************
 * 
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_brightness(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    void *brightness_args = gpuimage_brightness_args(args);
    if (brightness_args == NULL)
    {
        return NULL;
    }

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_brightness(obj_data, brightness_args);

    free(brightness_args);
    return Py_None;
}


/*******************************************************************************
 * Create a clone of the given image. All memory will be identical -- a deep
 * copy is performed.
 * 
 * Note that cloning won't necessarily place the new copy on the same device.
 * Specify target device before calling clone to ensure the copy is on a
 * specific device.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return A new reference that is a copy of self
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_clone(PyGPUImageObject *self, void *closure)
{
    int device_id;
    device_id = mpdev_get_target_device();
    if (device_id == DEVICE_LOC_NO_AFFINITY)
    {
        device_id = mpdev_get_recommended_device();
    }

    return gpuimage_clone(self, device_id, 0);
}


/*******************************************************************************
 * 
 * @return A new refrerence that is a copy of self
 * 
 ******************************************************************************/
PyObject *
gpuimage_clone(PyGPUImageObject *self, int device_id, int stream_id)
{
    PyObject *mp_module = PyImport_ImportModule("millipyde");
    PyTypeObject *gpuimage_type =
        (PyTypeObject *)PyObject_GetAttrString(mp_module, "gpuimage");

    // New returns a new reference to an image we now own. Don't increment ref count 
    PyObject *gpuimage = _PyObject_New(gpuimage_type);

    ((PyGPUImageObject *)gpuimage)->array.obj_data =
        mpobj_clone_data(self->array.obj_data, device_id, stream_id);

    Py_DECREF(mp_module);
    Py_DECREF(gpuimage_type);
    return gpuimage;
}


/*******************************************************************************
 * 
 * @return A new refrerence to a gpuimage
 * 
 ******************************************************************************/
PyObject *
gpuimage_single_from_path(PyObject *path)
{
    PyObject *args;

    PyObject *skimage_module = PyImport_ImportModule("skimage.io");
    if (skimage_module == NULL)
    {
        return NULL;
    }

    PyObject *mp_module = PyImport_ImportModule("millipyde");
    if (mp_module == NULL)
    {
        Py_DECREF(skimage_module);
        return NULL;
    }

    PyObject *conversion_func = PyObject_GetAttrString(skimage_module, (char *)"imread");
    if (conversion_func == NULL)
    {
        Py_DECREF(skimage_module);
        Py_DECREF(mp_module);
        return NULL;
    }

    // Use scikit image to generate an ndimage from the image path
    args = PyTuple_Pack(1, path);
    PyObject *image = PyObject_CallObject(conversion_func, args);
    Py_DECREF(args);

    // We are done using the conversion function
    Py_DECREF(conversion_func);

    PyTypeObject *gpuarray_type =
        (PyTypeObject *)PyObject_GetAttrString(mp_module, "gpuimage");
    if (gpuarray_type == NULL)
    {
        Py_DECREF(skimage_module);
        Py_DECREF(mp_module);
        Py_DECREF(image);
        return NULL;
    }

    // New returns a new reference to the array we now own. Don't increment ref count 
    PyObject *gpuarray = _PyObject_New(gpuarray_type);
    
    args = PyTuple_Pack(1, image);
    PyGPUImage_init((PyGPUImageObject *)gpuarray, PyTuple_Pack(1, image), NULL);
    
    Py_DECREF(args);
    Py_DECREF(image);
    Py_DECREF(skimage_module);
    Py_DECREF(mp_module);
    Py_DECREF(gpuarray_type);
    
    return gpuarray;
}


/*******************************************************************************
 * 
 * @return A new refrerence to a list containing gpuimages
 * 
 ******************************************************************************/
PyObject *
gpuimage_all_from_path(PyObject *path)
{
    // Returns a new reference that we now own. Size must start as 0.
    PyObject *result_list = PyList_New(0);

    const char *path_name = PyUnicode_AsUTF8(path);
    int path_len = PyUnicode_GET_LENGTH(path);

    int file_name_len;

    char full_path[256];

    memcpy(full_path, path_name, path_len);
    if (full_path[path_len - 1] != '/')
    {
        full_path[path_len] = '/';
        ++path_len;
    }


    DIR *dir;
    struct dirent *entry;
    dir = opendir(path_name);

    if (dir)
    {   
        // Read all directories in the path location
        while ((entry = readdir(dir)) != NULL)
        {
            // Make sure the entry is a file
            if (entry->d_type == DT_REG && valid_image_filename(entry->d_name))
            {
                file_name_len = _D_EXACT_NAMLEN(entry);
                memcpy(full_path + path_len, entry->d_name, file_name_len);
                full_path[path_len + file_name_len] = '\0';

                PyObject *image = gpuimage_single_from_path(PyUnicode_FromString(full_path));

                PyList_Append(result_list, image);

                // Adding to the list increments the ref count. We don't need our reference
                Py_DECREF(image);

            }
        }
        closedir(dir);
    }
    return result_list;
}


void *
gpuimage_rotate_args(PyObject *args)
{
    double angle;
    RotateArgs *rotate_args = malloc(sizeof(RotateArgs));
    if (!PyArg_ParseTuple(args, "d", &angle))
    {
        return NULL;
    }
    rotate_args->angle = angle;
    return (void *)rotate_args;
}


void *
gpuimage_gaussian_args(PyObject *args)
{
    double sigma;
    GaussianArgs *gaussian_args = malloc(sizeof(GaussianArgs));
    if (!PyArg_ParseTuple(args, "d", &sigma))
    {
        return NULL;
    }
    gaussian_args->sigma = sigma;
    return (void *)gaussian_args;
}


void *
gpuimage_brightness_args(PyObject *args)
{
    double delta;
    
    if (!PyArg_ParseTuple(args, "d", &delta))
    {
        return NULL;
    }

    if (delta <= -1 || delta >= 1)
    {
        return NULL;
    }

    BrightnessArgs *brightness_args = malloc(sizeof(BrightnessArgs));
    brightness_args->delta = delta;
    return (void *)brightness_args;
}


MPBool
gpuimage_check(PyObject *object)
{
    MPBool ret_val = MP_FALSE;
    PyObject *mp_module = PyImport_ImportModule("millipyde");
    PyTypeObject *gpuimage_type =
        (PyTypeObject *)PyObject_GetAttrString(mp_module, "gpuimage");

    if (object->ob_type == gpuimage_type)
    {
        ret_val = MP_TRUE;
    }

    Py_DECREF(mp_module);
    Py_DECREF(gpuimage_type);
    return ret_val;
}


MPBool
valid_image_filename(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if(!ext || ext == filename)
    {
        // dot at beginning of a filename has special meaning. Not related to extension
        return MP_FALSE;
    }

    if ((0 == strcasecmp(ext + 1, "png")) ||
        (0 == strcasecmp(ext + 1, "jpg")) ||
        (0 == strcasecmp(ext + 1, "jpeg")) ||
        (0 == strcasecmp(ext + 1, "tiff")) ||
        (0 == strcasecmp(ext + 1, "bmp")))
    {
        return MP_TRUE;
    }

    return MP_FALSE;
}
