from setuptools import setup, find_packages

setup(name='las',
      version='0.1.0',
      install_requires=['tensorflow==2.6.*', 'torch==1.8.2+cu111'],
      packages=find_packages())
