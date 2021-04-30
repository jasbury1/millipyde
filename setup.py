from distutils.core import setup, Extension
import numpy

# define the extension module
millipyde_module = Extension('millipyde', sources=['src/millipyde_module.c'],
                          include_dirs=[numpy.get_include(), 'include/'])

# run the setup
setup(ext_modules=[millipyde_module])