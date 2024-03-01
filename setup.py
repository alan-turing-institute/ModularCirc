#!/usr/bin/env python3
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
home_page = 'https://github.com/MaxBalmus/ModularCirc'

def read_requirements(file_name):
    reqs = []
    with open(os.path.join(here, file_name)) as in_f:
        for line in in_f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            reqs.append(line)
    return reqs


with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setup(
    name='ModularCirc', 
    version='0.1.0', 
    url=here,
    author='Maximilian Balmus',
    author_email='mbalmus@turing.ac.uk',
    license='MIT License',
    description='A python package for creating and running 0D models of the cardiovascular system', 
    long_description=readme,
    packages=find_packages(),
    install_requirements=read_requirements('requirements.txt'),
    python_requirements='>3.8',
    project_urls={
        "Bug Tracker": os.path.join(home_page, 'issues'),
        "Source Code": home_page,
    },
    extras_require={
        "dev" : read_requirements('requirements.txt')
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environemnt :: Console',
        'Intended Audience :: Researchers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10'
    ]
)