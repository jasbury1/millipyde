from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension
import pathlib
import numpy
import os

use_nvcc = False
if (os.environ['HIP_PLATFORM'] == 'nvidia'):
    use_nvcc = True

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

mp_sources = ['src/millipyde_module.c',
              'src/millipyde.c',
              'src/gpuarray.c',
              'src/gpuimage.c',
              'src/gpuoperation.c',
              'src/gpupipeline.c',
              'src/gpugenerator.c',
              'src/device.c',
              'src/millipyde_image.cpp',
              'src/millipyde_objects.cpp',
              'src/millipyde_devices.cpp',
              'src/millipyde_workers.cpp']

mp_include_dirs = [numpy.get_include(),
                   'src/include/',
                   '/opt/rocm-4.5.0/hip/include/hip']

if use_nvcc:
    compile_args = ['-DNDEBUG',
                    '-g',
                    '-O2',
                    '-D_FORTIFY_SOURCE=2',
                    '-Xcompiler',
                    '-fPIC']
    link_args = ['-shared',
                 '-g',
                 '-O2',
                 '-D_FORTIFY_SOURCE=2']
else:
    compile_args = []
    link_args = []


class nvcc_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so",
                                     "/opt/rocm-4.5.0/hip/bin/hipcc")
        self.compiler.set_executable("compiler_cxx", "/opt/rocm-4.5.0/hip/bin/hipcc")
        self.compiler.set_executable("linker_so", "/opt/rocm-4.5.0/hip/bin/hipcc")
        build_ext.build_extensions(self)


millipyde_module = Extension('millipyde',
                             sources=mp_sources,
                             include_dirs=mp_include_dirs,
                             extra_compile_args=compile_args,
                             extra_link_args=link_args,
                             # Cython uses old numpy versions. Disable warning until fixed upstream.
                             define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

os.environ["CC"] = "/opt/rocm-4.5.0/hip/bin/hipcc"
os.environ["CXX"] = "/opt/rocm-4.5.0/hip/bin/hipcc"

if use_nvcc:
    setup(
        version='0.0.1.dev1',
        description='A framework for GPU computing',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/jasbury1/millipyde',
        author='James Asbury',
        author_email='jasbury@calpoly.edu',

        keywords='gpu, parallel, array, rocm, hip, amd, framework, image, augmentation',

        ext_modules=[millipyde_module],
        cmdclass={"build_ext": nvcc_build_ext},

        project_urls={
            'Bug Reports': 'https://github.com/jasbury1/millipyde/issues',
            'Source': 'https://github.com/jasbury1/millipyde',
        },
    )
else:
    setup(
        version='0.0.1.dev1',
        description='A framework for GPU computing',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/jasbury1/millipyde',
        author='James Asbury',
        author_email='jasbury@calpoly.edu',

        keywords='gpu, parallel, array, rocm, hip, amd, framework, image, augmentation',

        ext_modules=[millipyde_module],

        project_urls={
            'Bug Reports': 'https://github.com/jasbury1/millipyde/issues',
            'Source': 'https://github.com/jasbury1/millipyde',
        },
    ) 
