# Categorical DQN (C-51)
Pytorch implementation of the Categorical DQN algorithm proposed in [A Distributional 
Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887).

## Run
To run the algorithm just run the `categorical_dqn.py` script. Usage:
```commandline
usage: categorical_dqn.py [-h] [--env ENV] [--n-atoms N_ATOMS]
                          [--n-hidden-units N_HIDDEN_UNITS]
                          [--n-hidden-layers N_HIDDEN_LAYERS]
                          [--support-range SUPPORT_RANGE]
                          [--start-train-at START_TRAIN_AT]
                          [--update-net-every UPDATE_NET_EVERY]
                          [--epsilon EPSILON] [--n-steps N_STEPS]
                          [--out-file OUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Gym environment. Currently supports only Box
                        observation spaces and Discrete action spaces.
                        Defaults to CartPole-v0.
  --n-atoms N_ATOMS     Number of atoms in the distributional network output.
                        Defaults to 51.
  --n-hidden-units N_HIDDEN_UNITS
                        Number of hidden units for each layer of the
                        distributional network. Defaults to 64.
  --n-hidden-layers N_HIDDEN_LAYERS
                        Number of hidden layers in the distributional network.
                        Defaults to 2.
  --support-range SUPPORT_RANGE
                        Range of the support of rewards. Ideally, it should be
                        [min, max], where min and max are referred to the
                        min/max cumulative discounted reward obtainable in one
                        episode. Defaults to [0, 200].
  --start-train-at START_TRAIN_AT
                        How many steps to collect before starting training.
                        Defaults to 32.
  --update-net-every UPDATE_NET_EVERY
                        How often to update the target network. Defaults to 5.
  --epsilon EPSILON     Exploration noise. Defaults to 0.1.
  --n-steps N_STEPS     Number of training steps. Defaults to 2000.
  --out-file OUT_FILE   If specified, stores the training plot into the given
                        file. The plot is stored as a json object with the
                        keys 'steps' and 'rewards'.

```