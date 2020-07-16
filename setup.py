#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages


__version__ = '0.1.0'
url = 'https://github.com/EdisonLeeeee/Graphgallery'

install_requires = [
    'tensorflow>=2.1',
    'numpy',
    'tqdm',
    'scipy',
    'networkx==2.4',
    'scikit-learn',
    'numba',
    'requests',
    'pandas',
    'rdflib',
    'h5py',
    'googledrivedownloader',
    'metis==0.2a4',
    'gensim',
    'texttable'
]

setup(
    name='graphgallery',
    version=__version__,
    description='Geometric Deep Learning Extension Library for TensorFlow',
    author='Jintang Li',
    author_email='cnljt@outlook.com',
    long_description=open("README.md", encoding="utf-8").read(),
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'tensorflow',
        'geometric-deep-learning',
        'graph-neural-networks',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages()
)
