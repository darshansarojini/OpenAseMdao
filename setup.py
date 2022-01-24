#!/usr/bin/env python

from setuptools import setup

setup(
    name="open_ase_mdao",
    version="0.1",
    description="Aero-servo-elastic Multi-disciplinary design optimization",
    author=["Darshan Sarojini", "David Solano"],
    author_email=["darshan.sheshgiri@gmail.com", "hd.solano@gmail.com"],
    packages=["open_ase_mdao"],
    install_requires=[
        'h5py',
        'numpy',
        'pint',
        'scipy',
        'sympy',
        'pandas',
        'plotly',
        'openmdao',
        'openaerostruct'
    ],
)
