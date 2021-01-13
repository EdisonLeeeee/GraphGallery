#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from os import path


# From: https://github.com/facebookresearch/iopath/blob/master/setup.py
# Author: Facebook Research
def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "graphgallery", "version.py"
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
    'tqdm',
    'yacs',
    'scipy',
    'numpy',
    'tabulate',
    'scikit_learn',
    'torch>=1.4',
    'tensorflow>=2.1.0',
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
    author_email='cnljt@outlook.com',
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
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        "Operating System :: OS Independent"
    ],
)
