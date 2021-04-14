import torch
from torch import nn
from torch.nn import functional as F


class LinearNetwork(nn.Module):
    """A network composed of several linear layers.
    """

    def __init__(self, inputs, outputs, n_hidden_layers, n_hidden_units, activation=torch.relu,
                 activation_last_layer=None, output_weight=1., dtype=torch.float):
        """Create a linear neural network with the given number of layers and units and
        the given activations.
        Args:
            inputs (int): Number of input nodes.
            outputs (int): Number of output nodes.
            n_hidden_layers (int): Number of hidden layers, excluding input and output layers.
            n_hidden_units (int): Number of units in the hidden layers.
            activation: The activation function that will be used in all but the last layer. Use
                None for no activation.
            activation_last_layer: The activation function to be used in the last layer. Use None
                for no activation.
            output_weight (float): Weight(s) to multiply to the last layer output.
            dtype (torch.dtype): Type of the network weights.
        """
        super().__init__()
        self.activation = activation
        self.activation_last_layer = activation_last_layer
        self.output_weight = output_weight
        self.lin = nn.Linear(in_features=inputs, out_features=n_hidden_units)
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.hidden_layers.append(
                nn.Linear(
                    in_features=n_hidden_units,
                    out_features=n_hidden_units
                )
            )
        self.lout = nn.Linear(in_features=n_hidden_units, out_features=outputs)
        self.type(dtype)

    def forward(self, *inputs):
        """Forward pass on the concatenation of the given inputs.
        """
        cat_inputs = torch.cat([*inputs], 1)
        x = self.lin(cat_inputs)
        if self.activation is not None:
            x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.lout(x)
        if self.activation_last_layer is not None:
            x = self.activation_last_layer(x) * self.output_weight
        return x


class DiscreteActorCriticSplit(nn.Module):
    """Wrapper class that keeps discrete actor and critic as separate networks.
    """

    def __init__(self, actor, critic, add_softmax=True):
        """Instantiate the ActorCriticSplit from two networks.

        Args:
            actor (torch.nn.Module): Network that takes states as input and outputs the action
                distribution.
            critic (torch.nn.Module): Network that takes states as input and returns the
                values for each action.
            add_softmax (bool): Whether to add a softmax layer to the actor network.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.add_softmax = add_softmax

    def forward(self, states):
        """Compute action distribution and values for the given states.

        Args:
            states (torch.Tensor): A batch of N states.

        Returns:
            A tuple (action_probabilities, values) where both are tensors of shape (N, a_dim).
        """
        action_probabilities = self.actor(states)
        if self.add_softmax:
            action_probabilities = F.softmax(action_probabilities, dim=1)
        values = self.critic(states)
        return action_probabilities, values

    def no_grads(self):
        for act_p, crit_p in zip(self.actor.parameters(), self.critic.parameters()):
            act_p.requires_grad = False
            crit_p.requires_grad = False

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()


class DistributionalNetwork(LinearNetwork):

    def __init__(self, inputs, n_actions, n_atoms, n_hidden_layers, n_hidden_units,
                 activation=torch.relu, dtype=torch.float):
        super(DistributionalNetwork, self).__init__(inputs=inputs, outputs=n_actions*n_atoms,
                                                    n_hidden_layers=n_hidden_layers,
                                                    n_hidden_units=n_hidden_units,
                                                    activation=activation,
                                                    dtype=dtype)
        self.n_actions = n_actions
        self.n_atoms = n_atoms

    def forward(self, *inputs):
        x = super(DistributionalNetwork, self).forward(*inputs)
        x = x.reshape(x.shape[0], self.n_actions, self.n_atoms)
        x = nn.functional.softmax(x, dim=-1)
        return x
