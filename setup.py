#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from graphgallery import __version__

VERSION = __version__
url = 'https://github.com/EdisonLeeeee/GraphGallery'

install_requires = [
            'networkx==2.3',
            'metis==0.2a4',
            'scipy>=1.4.1',
            'tensorflow>=2.1.0',
            'numpy>=1.17.4',
            'gensim>=3.8.0',
            'texttable>=1.6.2',
            'numba==0.46.0',
            'llvmlite==0.30',
            'tqdm>=4.40.2',
            'scikit_learn>=0.22',
]

setup(
    name='graphgallery',
    version=VERSION,
    description='Geometric Deep Learning Extension Library for TensorFlow',
    author='Jintang Li',
    author_email='cnljt@outlook.com',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, VERSION),
    keywords=[
        'tensorflow',
        'geometric-deep-learning',
        'graph-neural-networks',
    ],
    python_requires='>=3.6',
    license="MIT LICENSE",    
    install_requires=install_requires,
    packages=setuptools.find_packages(exclude=("examples", "imgs")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
