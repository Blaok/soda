"""Stencil with Optimized Dataflow Architecture.

A stencil compiler that takes a high-level domain-specific language (DSL) as
input, applies parallelization, communication reuse, and computation reuse, and
generate optimized FPGA accelerators.

See:
https://github.com/Blaok/soda
"""

from setuptools import find_packages, setup

with open('../README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(
    name='sodac',
    version='0.0.20200601.dev1',
    description='Stencil with optimized dataflow architecture',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Blaok/soda',
    author='Blaok Chi',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: System :: Hardware',
    ],
    packages=find_packages(exclude=('tests', 'tests.*')),
    python_requires='>=3.6',
    install_requires=[
        'cached_property',
        'haoda>=0.0.20200505.dev1',
        'pulp',
        'textx',
        'toposort',
    ],
    entry_points={
        'console_scripts': ['sodac=soda.sodac:main'],
    },
)
