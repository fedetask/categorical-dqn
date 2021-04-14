import argparse
import copy
import random
import tqdm
import json

import gym
import torch

import memory
from memory import Transition
import networks
from utils import np_to_unsq_tensor, squeeze_np
from experiment_utils import Plot


def extract(transitions):
    """Extract tensors of s, a, r, s' from a batch of transitions.

    Args:
        transitions (list): List of Transition named tuples where next_state is None if episode
            ended.

    Returns:
        (states, actions, rewards, next_states, mask) that are all (batch_size, *shape) tensors
        containing the extracted data. next_states does not contain elements for episode that
        ended. mask is a boolean tensor that specifies which transitions have a next state.
    """
    states = torch.cat([t.state for t in transitions])
    actions = torch.cat([t.action for t in transitions])
    rewards = torch.cat([t.reward for t in transitions])
    mask = torch.tensor([t.next_state is not None for t in transitions])
    next_states = torch.cat([t.next_state for t in transitions if t.next_state is not None])
    return states, actions, rewards, next_states, mask


def select_argmax_action(z, atoms):
    # Take state-action distribution z, which is a (batch_size, action_size, n_atoms) and
    # returns a tensor of shape (batch_size, 1) with the greedy actions for each state
    q_values = (z * atoms[:, None, :]).sum(dim=-1)
    return q_values.argmax(dim=-1).unsqueeze(1)


class CategoricalDQN:

    def __init__(self, z_net, n_atoms, v_min, v_max, df=0.99, buffer_len=1e6, batch_size=32,
                 lr=0.5e-3, update_mode='hard', update_every=5, tau=0.05, epsilon=0.1,
                 start_train_at=4000, results_dir=None):
        self.z_net = z_net
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta = (v_max - v_min) / n_atoms
        self.df = df
        self.buffer_len = buffer_len
        self.batch_size = batch_size
        self.update_mode = update_mode
        self.update_every = update_every
        self.tau = tau
        self.epsilon = epsilon
        self.start_train_at = start_train_at
        self.replay_buffer = memory.TransitionReplayBuffer(maxlen=buffer_len)
        self._target_net = copy.deepcopy(z_net)
        self.optimizer = torch.optim.Adam(self.z_net.parameters(), lr=lr)
        self.atoms = torch.arange(self.v_min, self.v_max, self.delta).unsqueeze(0)

    def train(self, env: gym.Env, n_steps):
        rewards = []
        steps = []
        episode_rewards = []
        state = np_to_unsq_tensor(env.reset())
        loop_range = tqdm.tqdm(range(n_steps))
        for step in loop_range:
            with torch.no_grad():
                z = self.z_net(state)
            if random.random() < self.epsilon:  # Random action
                action = torch.LongTensor([[env.action_space.sample()]])
            else:
                action = select_argmax_action(z, self.atoms)
            next_state, reward, done, info = env.step(squeeze_np(action))
            next_state = np_to_unsq_tensor(next_state) if not done else None
            self.replay_buffer.remember(
                Transition(state, action, torch.tensor([[reward]]), next_state))
            state = next_state

            # Perform training step
            self._train_step(step)

            # Update episode stats
            episode_rewards.append(reward)
            if done:
                state = np_to_unsq_tensor(env.reset())
                rewards.append(sum(episode_rewards))
                steps.append(step)
                episode_rewards = []
                loop_range.set_description(f'Reward {rewards[-1]}')
        return Plot(steps, rewards, None)

    def _train_step(self, step):
        if step < self.start_train_at or self.replay_buffer.size() < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, mask = extract(batch)
        targets = self._compute_targets(rewards, next_states, mask)
        self._train_net(states, actions, targets, update=(step % self.update_every) == 0)

    def _train_net(self, states, actions, targets, update):
        self.optimizer.zero_grad()
        z = self.z_net(states)
        z = torch.cat([z[i, actions[i]] for i in range(z.shape[0])])
        # Compute cross-entropy loss
        loss = -(targets * z.log()).sum(dim=-1).mean()
        loss.backward()
        self.optimizer.step()
        if update:
            self._update_target_net()

    def _update_target_net(self):
        # Mode can be 'hard' or 'soft'
        if self.update_mode == 'hard':
            self._target_net.load_state_dict(self.z_net.state_dict())
        else:
            for param, target_param in zip(self.z_net.parameters(), self._target_net.parameters()):
                target_param.copy_(self.tau * param + (1 - self.tau) * target_param)

    def _compute_targets(self, rewards, next_states, mask):
        """Compute the target distributions for the given transitions.

        """
        # All these are (batch_size, *shape) tensors
        atoms = torch.arange(self.v_min, self.v_max, self.delta)
        atoms = (rewards + self.df * mask[:, None] * atoms).clamp(min=self.v_min, max=self.v_max)
        b = (atoms - self.v_min) / self.delta
        l = torch.floor(b).long()
        u = torch.ceil(b).clamp(max=self.n_atoms - 1).long()  # Prevent out of bounds
        # Predict next state return distribution for each action
        with torch.no_grad():
            z_prime = self._target_net(next_states)
        target_actions = select_argmax_action(z_prime, atoms[mask])
        # TODO: Do this with gather or similar
        z_prime = torch.cat([z_prime[i, target_actions[i]] for i in range(z_prime.shape[0])])

        # For elements that do not have a next state, atoms are all equal to reward and we set a
        # uniform distribution (it will collapse to the same atom in any case)
        probabilities = torch.ones((self.batch_size, self.n_atoms)) / self.n_atoms
        probabilities[mask] = z_prime
        # Compute partitions of atoms
        lower = probabilities * (u - b)
        upper = probabilities * (b - l)
        z_projected = torch.zeros_like(probabilities)
        z_projected.scatter_add_(1, l, lower)
        z_projected.scatter_add_(1, u, upper)
        return z_projected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=False, default='CartPole-v0',
                        help='Gym environment. Currently supports only Box observation spaces '
                             'and Discrete action spaces. Defaults to CartPole-v0.')
    parser.add_argument('--n-atoms', type=int, required=False, default=51,
                        help='Number of atoms in the distributional network output. '
                             'Defaults to 51.')
    parser.add_argument('--n-hidden-units', type=int, required=False, default=64,
                        help='Number of hidden units for each layer of the distributional '
                             'network. Defaults to 64.')
    parser.add_argument('--n-hidden-layers', type=int, required=False, default=2,
                        help='Number of hidden layers in the distributional network. '
                             'Defaults to 2.')
    parser.add_argument('--support-range', type=list, required=False, default=[0, 200],
                        help='Range of the support of rewards. Ideally, it should be [min, max], '
                             'where min and max are referred to the min/max cumulative discounted '
                             'reward obtainable in one episode. Defaults to [0, 200].')
    parser.add_argument('--start-train-at', type=int, required=False, default=32,
                        help='How many steps to collect before starting training. '
                             'Defaults to 32.')
    parser.add_argument('--update-net-every', type=int, required=False, default=5,
                        help='How often to update the target network. Defaults to 5.')
    parser.add_argument('--epsilon', type=float, required=False, default=0.1,
                        help='Exploration noise. Defaults to 0.1.')
    parser.add_argument('--n-steps', type=int, required=False, default=20000,
                        help='Number of training steps. Defaults to 2000.')
    parser.add_argument('--out-file', type=str, required=False, default=None,
                        help='If specified, stores the training plot into the given file. The '
                             'plot is stored as a json object with the keys \'steps\' and '
                             '\'rewards\'.')
    args = parser.parse_args()

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    n_atoms = args.n_atoms
    n_units = args.n_hidden_units
    n_layers = args.n_hidden_layers
    z_net = networks.DistributionalNetwork(inputs=state_dim, n_actions=act_dim, n_atoms=n_atoms,
                                           n_hidden_units=n_units, n_hidden_layers=n_layers)
    v_min, v_max = args.support_range
    start_train_at = args.start_train_at
    update_net_every = args.update_net_every
    epsilon = args.epsilon
    n_steps = args.n_steps
    DDQN = CategoricalDQN(z_net=z_net, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                          start_train_at=start_train_at,
                          update_every=update_net_every, epsilon=epsilon)
    plot = DDQN.train(env=env, n_steps=n_steps)

    if args.out_file is not None:
        with open(args.out_file, 'w') as fp:
            json.dump({'steps': plot.x.tolist(), 'rewards': plot.y.tolist()}, fp)