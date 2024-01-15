#!/usr/bin/env python
import setuptools

VERSION = '1.0.0'

setuptools.setup(
    name='body_comp',
    version=VERSION,
    description='Package for training and deployment neural networks for body composition analysis of abdominal CTs',
    author='Christopher P. Bridge',
    maintainer='Christopher P. Bridge',
    url='https://gitlab.partners.org/mr118/ct_body_composition',
    platforms=['Linux'],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.16.0',
        'tensorflow>=2.3.0',
        'matplotlib>=3.2.0',
        'scikit-image>=0.16.0',
        'scipy>=1.4.0',
        'pydicom>=2.0.0',
        'pandas>=1.0.3',
        'highdicom>=0.21.0',
        'python-gdcm>=3.0.0',
    ],
    package_data={
        '': ['models/*.hdf5', 'configs/*.json'],
    }
)
