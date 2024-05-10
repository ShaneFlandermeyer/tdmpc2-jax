# tdmpc2-jax

A re-implementation of [TD-MPC2](https://www.tdmpc2.com/) in Jax/Flax. JIT'ing the planning/update steps makes training 5-10x faster compared to the original PyTorch implementation.

This repository also supports vectorized environments (see the env field of ```config.yaml```) and finite-horizon environments (see ```world_model.predict_continues``` and ```tdmpc.continue_coef``` in ```config.yaml```).

## Usage

To install the dependencies for this project (tested on Ubuntu 22.04), run

```[bash]
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install --upgrade tqdm numpy flax optax jaxtyping einops gymnasium[mujoco]
```

Then, edit ```config.yaml``` and run ```train.py``` in the main project directory.


## Installation

Install the package from the base directory with

```[bash]
pip install -e .
```
