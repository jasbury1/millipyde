#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>
#include <dirent.h>
#include <string.h>

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


PyObject *
PyGPUImage_rand_rotate(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    double min, max;
    if (!PyArg_ParseTuple(args, "dd", &min, &max))
    {
        return NULL;
    }

    double angle;
    random_double_in_range(min, max, &angle);

    PyObject *a = Py_BuildValue("d", angle);
    PyObject *arg = PyTuple_Pack(1, a);

    PyGPUImage_rotate(self, arg, NULL);

    Py_DECREF(a);
    Py_DECREF(arg);

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


PyObject *
PyGPUImage_rand_gaussian(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    double min, max;
    if (!PyArg_ParseTuple(args, "dd", &min, &max))
    {
        return NULL;
    }

    double sigma;
    random_double_in_range(min, max, &sigma);

    PyObject *a = Py_BuildValue("d", sigma);
    PyObject *arg = PyTuple_Pack(1, a);

    PyGPUImage_gaussian(self, arg, NULL);

    Py_DECREF(a);
    Py_DECREF(arg);

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


PyObject *
PyGPUImage_rand_brightness(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    double min, max;
    if (!PyArg_ParseTuple(args, "dd", &min, &max))
    {
        return NULL;
    }

    double delta;
    random_double_in_range(min, max, &delta);

    PyObject *a = Py_BuildValue("d", delta);
    PyObject *arg = PyTuple_Pack(1, a);

    PyGPUImage_brightness(self, arg, NULL);

    Py_DECREF(a);
    Py_DECREF(arg);

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
PyGPUImage_adjust_gamma(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    void *gamma_args = gpuimage_gamma_args(args);
    if (gamma_args == NULL)
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
    mpimg_adjust_gamma(obj_data, gamma_args);

    free(gamma_args);
    return Py_None;
}


PyObject *
PyGPUImage_rand_adjust_gamma(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *gamma_range;
    PyObject *gain_range;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &gamma_range, &PyList_Type,
                          &gain_range))
    {
        return NULL;
    }

    // TODO: Better type checking needed here
    double gamma_min, gamma_max, gain_min, gain_max;
    gamma_min = PyFloat_AsDouble(PyList_GetItem(gamma_range, 0));
    gamma_max = PyFloat_AsDouble(PyList_GetItem(gamma_range, 1));
    gain_min = PyFloat_AsDouble(PyList_GetItem(gain_range, 0));
    gain_max = PyFloat_AsDouble(PyList_GetItem(gain_range, 1));
    

    double gamma, gain;

    random_double_in_range(gamma_min, gamma_max, &gamma);
    random_double_in_range(gain_min, gain_max, &gain);

    PyObject *a1 = Py_BuildValue("d", gamma);
    PyObject *a2 = Py_BuildValue("d", gain);
    PyObject *arg = PyTuple_Pack(2, a1, a2);

    PyGPUImage_adjust_gamma(self, arg, NULL);

    Py_DECREF(a1);
    Py_DECREF(a2);
    Py_DECREF(arg);

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
PyGPUImage_colorize(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    void *colorize_args = gpuimage_colorize_args(args);
    if (colorize_args == NULL)
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
    mpimg_colorize(obj_data, colorize_args);

    free(colorize_args);
    return Py_None;
}


PyObject *
PyGPUImage_rand_colorize(PyGPUImageObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *r_range;
    PyObject *g_range;
    PyObject *b_range;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyList_Type, &r_range, &PyList_Type,
                          &g_range, &PyList_Type, &b_range))
    {
        return NULL;
    }

    // TODO: Better type checking needed here
    double r_min, r_max, g_min, g_max, b_min, b_max;
    r_min = PyFloat_AsDouble(PyList_GetItem(r_range, 0));
    r_max = PyFloat_AsDouble(PyList_GetItem(r_range, 1));
    g_min = PyFloat_AsDouble(PyList_GetItem(g_range, 0));
    g_max = PyFloat_AsDouble(PyList_GetItem(g_range, 1));
    b_min = PyFloat_AsDouble(PyList_GetItem(b_range, 0));
    b_max = PyFloat_AsDouble(PyList_GetItem(b_range, 1));

    double r_mult, g_mult, b_mult;

    random_double_in_range(r_min, r_max, &r_mult);
    random_double_in_range(g_min, g_max, &g_mult);
    random_double_in_range(b_min, b_max, &b_mult);

    PyObject *a1 = Py_BuildValue("d", r_mult);
    PyObject *a2 = Py_BuildValue("d", g_mult);
    PyObject *a3 = Py_BuildValue("d", b_mult);
    PyObject *arg = PyTuple_Pack(3, a1, a2, a3);

    PyGPUImage_colorize(self, arg, NULL);

    Py_DECREF(a1);
    Py_DECREF(a2);
    Py_DECREF(a3);
    Py_DECREF(arg);

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


void *
gpuimage_colorize_args(PyObject *args)
{
    double r_mult;
    double g_mult;
    double b_mult;
    
    if (!PyArg_ParseTuple(args, "ddd", &r_mult, &g_mult, &b_mult))
    {
        return NULL;
    }

    if (r_mult < 0 || g_mult < 0 || b_mult < 0)
    {
        return NULL;
    }

    ColorizeArgs *colorize_args = malloc(sizeof(ColorizeArgs));
    colorize_args->r_mult = r_mult;
    colorize_args->g_mult = g_mult;
    colorize_args->b_mult = b_mult;
    return (void *)colorize_args;
}


void *
gpuimage_gamma_args(PyObject *args)
{
    double gamma;
    double gain;

    if (!PyArg_ParseTuple(args, "dd", &gamma, &gain))
    {
        return NULL;
    }

    GammaArgs *gamma_args = malloc(sizeof(GammaArgs));
    gamma_args->gain = gain;
    gamma_args->gamma = gamma;

    return (void *)gamma_args;
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

