from setuptools import setup, find_packages, Extension
import pathlib
import numpy
import os

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

millipyde_module = Extension('millipyde',
                             sources=['src/millipyde_module.c', 
                                 'src/gpuarray.c',
                                 'src/gpuimage.c',
                                 'src/bit_extract.cpp',
                                 'src/millipyde_image.cpp',
                                 'src/gpuarray_funcs.cpp',
                                 'src/millipyde_devices.cpp'],
                             include_dirs=[numpy.get_include(), 
                                 'src/include/',
                                 '/opt/rocm-4.1.0/hip/include/hip'])

os.environ["CC"] = "/opt/rocm-4.1.0/hip/bin/hipcc"
os.environ["CXX"] = "/opt/rocm-4.1.0/hip/bin/hipcc"
                                

setup(
    version='0.0.1.dev1',
    description='A framework for GPU computing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jasbury1/millipyde',
    author='James Asbury',
    author_email='jasbury@calpoly.edu',

    keywords='gpu, parallel, array',

    #packages = ['millipyde', 'millipyde.img'],

    ext_modules=[millipyde_module],

    project_urls={
        'Bug Reports': 'https://github.com/jasbury1/millipyde/issues',
        'Source': 'https://github.com/jasbury1/millipyde',
    },
)
