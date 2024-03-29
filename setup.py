#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Gavin Gray",
    author_email='gngdb.labs@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Use scipy.optimize.minimize as a PyTorch Optimizer.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n',
    include_package_data=True,
    keywords='pytorch_minimize',
    name='pytorch_minimize',
    packages=find_packages(include=['pytorch_minimize', 'pytorch_minimize.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gngdb/pytorch_minimize',
    version='0.2.0',
    zip_safe=False,
)
