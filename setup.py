#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages

VERSION = "0.6.0"
url = 'https://github.com/EdisonLeeeee/GraphGallery'

install_requires = [
    'torch',
    'tqdm',
    'scipy',
    'numpy',
    'scikit_learn',
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
    description='A Gallery for Benchmarking Graph Neural Networks.',
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
    python_requires='>=3.7',
    license="MIT LICENSE",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    packages=find_packages(exclude=("examples", "imgs", "scripts",
                                    "benchmark")),
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
