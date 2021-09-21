#!/usr/bin/env python
# coding: utf-8
import torch
import os.path as osp
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None


def get_version():
    # From: https://github.com/facebookresearch/iopath/blob/master/setup.py
    # Author: Facebook Research
    init_py_path = osp.join(
        osp.abspath(osp.dirname(__file__)), "graphgallery", "version.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][
        0
    ]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


VERSION = get_version()
url = 'https://github.com/EdisonLeeeee/GraphGallery'


install_requires = [
    # Do not add pytorch and tensorflow here. User should install
    # pytorch and tensorflow themselves, preferrably by OS's package manager, or by
    # choosing the proper pypi package name at
    # pytorch: https://pytorch.org/
    # tensorflow: http://tensorflow.org/
    'tqdm',
    'yacs',
    'scipy',
    'numpy',
    'tabulate',
    'pandas',
    'scikit_learn>=0.21.0',
    'networkx>=2.3',
    'gensim>=3.8.0',
    'numba>=0.46.0',
]

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='graphgallery',
    version=VERSION,
    description='A Gallery for Benchmarking Graph Neural Networks and Graph Adversarial Learning.',
    author='Jintang Li',
    author_email='lijt55@mail2.sysu.edu.cn',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, VERSION),
    keywords=[
        'tensorflow',
        'pytorch',
        'benchmark',
        'geometric-deep-learning',
        'graph-neural-networks',
    ],
    python_requires='>=3.6',
    license="MIT LICENSE",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    packages=find_packages(exclude=("examples", "imgs", "benchmark", "test")),
    ext_modules=[
        CppExtension("graphgallery.sampler", sources=["csrc/cpu/neighbor_sampler_cpu.cpp"], extra_compile_args=['-g']),

    ],
    cmdclass={
        'build_ext':
        BuildExtension
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
