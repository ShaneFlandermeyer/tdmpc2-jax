from setuptools import setup, find_packages

setup(
    name='tdmpc2-jax',
    version='0.1.0',    
    description='Jax implementation of TD-MPC2',
    url='https://github.com/ShaneFlandermeyer/tdmpc2-jax',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'jax',
      'jaxlib',
      'tqdm',
      'numpy',
      'flax',
      'optax',
      'jaxtyping',
      'einops',
      'gymnasium[mujoco]'                     
    ],

)