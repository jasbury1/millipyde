from setuptools import setup, find_packages, Extension
import pathlib
import numpy

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

# define the extension module
millipyde_module = Extension(
    'millipyde', 
    sources=['src/millipyde_module.c'],
    include_dirs=[numpy.get_include(), 'src/include/']
    )

# run the setup
#setup(
#    ext_modules=[millipyde_module])

setup(
    name='millipyde',
    version='0.0.1.dev1',
    description='A framework for GPU computing',
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/jasbury1/millipyde',
    author='James Asbury',  # Optional
    author_email='jasbury@calpoly.edu',

    keywords='gpu, parallel, array',  # Optional

    ext_modules=[millipyde_module],
    
    package_dir={'': 'src'},

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(where='src'),  # Required

    python_requires='>=3.6, <4',

    install_requires=['numpy'],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jasbury1/millipyde/issues',
        'Source': 'https://github.com/jasbury1/millipyde',
    },
)