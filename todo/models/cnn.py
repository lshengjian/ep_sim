from torch import nn


class Flatten(nn.Module):
    """Helper to flatten a tensor."""

    def forward(self, x):
        return x.view(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class CNNBase(nn.Module):
    """CNN model."""

    def __init__(self, num_channels, num_outputs, dist, hidden_size=512):
        """Initializer.
            num_channels: the number of channels in the input images (eg 3
                for RGB images, or 12 for a stack of 4 RGB images).
            num_outputs: the dimension of the output distribution.
            dist: the output distribution (eg Discrete or Normal).
            hidden_size: the size of the final actor+critic linear layers
        """
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, \
            lambda x: nn.init.constant_(x, 0), \
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(), init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(), init_(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            nn.ReLU(), Flatten(), init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.actor_linear = init_(nn.Linear(hidden_size, num_outputs))
        self.dist = dist

    def forward(self, x):
        """x should have shape (batch_size, num_channels, 84, 84)."""
        x = self.main(x)
        value = self.critic_linear(x)
        action_logits = self.actor_linear(x)
        return value, self.dist(action_logits)