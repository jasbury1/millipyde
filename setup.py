from setuptools import setup, find_packages, Extension
import pathlib
import numpy

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

millipyde_module = Extension('millipyde',
                             sources=[
                                 'src/ndgpuarray.c',
                                 'src/millipyde_module.c'
                             ],
                             include_dirs=[numpy.get_include(), 'src/include/'])

setup(
    version='0.0.1.dev1',
    description='A framework for GPU computing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jasbury1/millipyde',
    author='James Asbury',  # Optional
    author_email='jasbury@calpoly.edu',

    keywords='gpu, parallel, array',

    ext_modules=[millipyde_module],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jasbury1/millipyde/issues',
        'Source': 'https://github.com/jasbury1/millipyde',
    },
)
