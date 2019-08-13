from setuptools import setup
from setuptools import find_packages

setup(name='gym_2048',
      version='0.0.1',
      description='2048 environment for OpenAi gym',
	author='Tim Übelhör',
	author_email='tim.uebelhoer@outlook.de',
	url='https://github.com/Tuebel/2048-gym',
      install_requires=['gym', 'numpy'],
      packages=find_packages())
