import wandb
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import math
import torch.optim as optim
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)


def lognorm_cdf_loss(moments, target):
    loss = torch.mean((moments - target * 2) ** 3)
    return loss


def gld_quantile(quantiles, l1, l2, l3, l4):
    """Compute the values of specified quantile levels from a generalized
    lambda distribution described by its lambda parameters.

    Args:
        quantiles (torch.tensor): set of quantiles to be evaluated
        l1 (torch.tensor): Location parameter of the GLD
        l2 (torch.tensor): Scale parameter of the GLD
        l3 (torch.tensor): First shape parameter of the GLD
        l4 (torch.tensor): Second shape parameter of the GLD

    Returns:
        torch.tensor: Values of the quantiles according to the desired GLD
    """
    a = (quantiles ** l3.clamp(-50, 50)).clamp(-1000, 1000)
    b = ((1 - quantiles) ** l4.clamp(-50, 50)).clamp(-1000, 1000)
    pos = torch.sign(l2) * 0.1
    return (l1 + (1 / (l2 + pos)) * (a - b)).clamp(-1000, 1000)


def pinball_loss(quantiles, predicted_values, actual_value):
    """Function to compute the pinball (or quantile) loss
    between a predicted value and an actual realization

    Args:
        quantiles (torch.tensor): quantiles to be evaluated
        predicted_values (torch.tensor): Predictions of quantile values made by a model
        actual_value (torch.tensor): realizations of the random variable

    Returns:
        torch.tensor: _description_
    """
    errors = [
        (actual_value - predicted_values) * q + F.relu(predicted_values - actual_value)
        for q in quantiles
    ]
    return (sum(errors) / len(quantiles)).mean()


class PinballLoss(nn.Module):
    """Class representing the pinball loss used in quantile regression."""

    def __init__(self, quantiles):
        """Creation of an instance of the class

        Args:
            quantiles (torch.tensor): quantiles to be used in the loss when evaluating an input and a target.
        """
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles
        self.size = len(quantiles)

    def forward(self, input, target):
        return pinball_loss(self.quantiles, input, target)


class PinballHuber(nn.Module):
    """Huber version of the pinball loss: it is an L2 around 0, instead of an L1."""

    def __init__(self, quantiles, delta=1.0):
        """
        Args:
            quantiles (torch.tensor): quantiles to be evaluated in the loss
            delta (float, optional): _description_. Defaults to 1.0.
        """
        super(PinballHuber, self).__init__()
        self.delta = delta
        self.quantiles = quantiles
        self.size = len(quantiles)
        self.quantile_mask = quantiles.repeat(self.size)

    def forward(self, policy_quantiles, target_quantiles):
        """Compuration of the huber pinball loss.

        Args:
            policy_quantiles (torch.tensor): quantile values computed for the current policy
            target_quantiles (torch.tensor): quantile values given by the target net

        Returns:
            torch.tensor: value of the Smoothed Pinball loss.   
        """
        target_quantiles_expanded = target_quantiles.repeat(1, self.size)
        policy_quantiles_expanded = policy_quantiles.repeat_interleave(self.size, 1)
        indic = (target_quantiles_expanded > policy_quantiles_expanded).type(
            torch.uint8
        )
        multiplier = torch.abs(self.quantile_mask - indic)
        huber = (
            F.huber_loss(
                target_quantiles_expanded,
                policy_quantiles_expanded,
                delta=self.delta,
                reduction="none",
            )
            / self.delta
        )
        return (multiplier * huber).mean().double()


class ExpectileLoss(nn.Module):
    def __init__(self, expectiles):
        super(ExpectileLoss, self).__init__()
        self.expectiles = expectiles
        self.size = len(expectiles)
        self.expectile_mask = expectiles.repeat(self.size)

    def forward(self, policy_expectiles, target_expectiles):
        target_expectiles_expanded = target_expectiles.repeat(1, self.size)
        policy_expectiles_expanded = policy_expectiles.repeat_interleave(self.size, 1)
        indic = (target_expectiles_expanded > policy_expectiles_expanded).type(
            torch.uint8
        )
        multiplier = torch.abs(self.expectile_mask - indic)
        squared_loss = (target_expectiles_expanded - policy_expectiles_expanded) ** 2
        return (multiplier * squared_loss).mean().double()


def cross_loss(m, probabilities):
    return (m * torch.log(probabilities)).neg().sum()


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # args = state, action, next_state, reward
        self.memory.extendleft([Transition(*arg) for arg in zip(*args)])

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DualNet(nn.Module):
    def __init__(self, model_args, name, device):
        super().__init__()
        net_type = globals()[name]
        self.policy_net = net_type(**model_args).to(device)
        self.target_net = net_type(**model_args).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def forward(self, o, net):
        if net == "policy":
            return self.policy_net(o)
        else:
            return self.target_net(o)


class DQN(nn.Module):
    def __init__(
        self,
        input_size,
        conv_size,
        hidden_sizes,
        n_kernels,
        output_size=251,
        quantiles=None,
        nonlinearity=torch.nn.SELU,
    ):
        super().__init__()
        self._conv_size = conv_size
        self._char_size = input_size - conv_size
        self.normL = torch.nn.LayerNorm(input_size)
        self._transf_in_size = input_size - conv_size + n_kernels
        self.activ = nn.SELU()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip(
                [self._transf_in_size] + hidden_sizes[:-1], hidden_sizes
            )
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend(
                [layer, nonlinearity(), torch.nn.LayerNorm(layer.out_features)]
            )
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.convs = torch.nn.Conv1d(1, out_channels=n_kernels, kernel_size=1)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, o):
        # oprime = self.normL(o)
        stock, characteristics = o.split([self._conv_size, self._char_size], -1)
        stock_summary = self.convs(stock.view(-1, 1, self._conv_size)).mean(2)
        summarized_input = torch.cat((self.activ(stock_summary), characteristics), -1)
        out = self.model(summarized_input)
        return out

    @property
    def output_size(self):
        return self._output_size


class ParameterDQN(nn.Module):
    def __init__(
        self,
        input_size,
        conv_size,
        hidden_sizes,
        n_kernels,
        output_size=251,
        quantiles=None,
        nonlinearity=torch.nn.SELU,
    ):
        super().__init__()
        self._conv_size = conv_size
        self._char_size = input_size - conv_size
        self.normL = torch.nn.LayerNorm(input_size)
        self._transf_in_size = input_size - conv_size + 3 * n_kernels
        self.activ = nn.SELU()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip(
                [self._transf_in_size] + hidden_sizes[:-1], hidden_sizes
            )
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend(
                [layer, nonlinearity(), torch.nn.LayerNorm(layer.out_features)]
            )
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        sequence.append(torch.nn.Linear(last_size, output_size))
        self.convs = torch.nn.Conv1d(1, out_channels=n_kernels, kernel_size=1)
        # self.model = torch.nn.Sequential(*sequence)
        self.l1_layer = torch.nn.Sequential(*sequence)
        self.l2_layer = torch.nn.Sequential(*sequence)
        self.l3_layer = torch.nn.Sequential(*sequence)
        self.l4_layer = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, o):
        stock, characteristics = o.split([self._conv_size, self._char_size], -1)
        conv_res = self.convs(stock.view(-1, 1, self._conv_size))
        stock_summary = torch.cat(
            (conv_res.mean(2), conv_res.max(2)[0], conv_res.min(2)[0]), 1
        )
        summarized_input = torch.cat((self.activ(stock_summary), characteristics), -1)
        lambda_one = self.l1_layer(summarized_input)
        lambda_two = self.l2_layer(summarized_input)
        lambda_three = self.l3_layer(summarized_input)
        lambda_four = self.l4_layer(summarized_input)
        return lambda_one, lambda_two, lambda_three, lambda_four

    @property
    def output_size(self):
        return self._output_size


class C51(nn.Module):
    def __init__(
        self,
        input_size,
        conv_size,
        hidden_sizes,
        n_kernels,
        output_size=251,
        quantiles=None,
        nonlinearity=torch.nn.SELU,
        n_atoms=51,
    ):
        super().__init__()
        self._conv_size = conv_size
        self._char_size = input_size - conv_size
        self.normL = torch.nn.LayerNorm(input_size)
        self._transf_in_size = input_size - conv_size + n_kernels
        self.activ = nn.SELU()
        self.n_atoms = n_atoms
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip(
                [self._transf_in_size] + hidden_sizes[:-1], hidden_sizes
            )
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend(
                [layer, nonlinearity(), torch.nn.LayerNorm(layer.out_features)]
            )
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        sequence.append(torch.nn.Linear(last_size, self.n_atoms * output_size))
        self.convs = torch.nn.Conv1d(1, out_channels=n_kernels, kernel_size=1)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, o):
        oprime = self.normL(o)
        stock, characteristics = o.split([self._conv_size, self._char_size], -1)
        stock_summary = self.convs(stock.view(-1, 1, self._conv_size)).mean(2)
        summarized_input = torch.cat((self.activ(stock_summary), characteristics), -1)
        res = self.model(summarized_input)
        return F.softmax(res.view(-1, self._output_size, self.n_atoms), dim=2)

    @property
    def output_size(self):
        return self._output_size


class QRDQN(nn.Module):
    def __init__(
        self,
        input_size,
        conv_size,
        hidden_sizes,
        n_kernels,
        output_size=251,
        nonlinearity=torch.nn.SELU,
        quantiles=[0.01, 0.25, 0.5, 0.75, 0.99],
    ):
        super().__init__()
        self._conv_size = conv_size
        self._char_size = input_size - conv_size
        self.normL = torch.nn.LayerNorm(input_size)
        self._transf_in_size = input_size - conv_size + n_kernels
        self.activ = nn.SELU()
        self.quantiles = quantiles
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip(
                [self._transf_in_size] + hidden_sizes[:-1], hidden_sizes
            )
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend(
                [layer, nonlinearity(), torch.nn.LayerNorm(layer.out_features)]
            )
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        sequence.append(torch.nn.Linear(last_size, len(self.quantiles) * output_size))
        self.convs = torch.nn.Conv1d(1, out_channels=n_kernels, kernel_size=1)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, o):
        oprime = self.normL(o)
        stock, characteristics = o.split([self._conv_size, self._char_size], -1)
        stock_summary = self.convs(stock.view(-1, 1, self._conv_size)).mean(2)
        summarized_input = torch.cat((self.activ(stock_summary), characteristics), -1)
        res = self.model(summarized_input)
        return res.view(-1, self._output_size, len(self.quantiles))

    @property
    def output_size(self):
        return self._output_size


class ERDQN(nn.Module):
    def __init__(
        self,
        input_size,
        conv_size,
        hidden_sizes,
        n_kernels,
        output_size=251,
        nonlinearity=torch.nn.SELU,
        quantiles=[0.01, 0.25, 0.5, 0.75, 0.99],
    ):
        super().__init__()
        self._conv_size = conv_size
        self._char_size = input_size - conv_size
        self.normL = torch.nn.LayerNorm(input_size)
        self._transf_in_size = input_size - conv_size + n_kernels
        self.activ = nn.SELU()
        self.expectiles = quantiles
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [
            torch.nn.Linear(n_in, n_out)
            for n_in, n_out in zip(
                [self._transf_in_size] + hidden_sizes[:-1], hidden_sizes
            )
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend(
                [layer, nonlinearity(), torch.nn.LayerNorm(layer.out_features)]
            )
        last_size = hidden_sizes[-1] if hidden_sizes else input_size
        sequence.append(torch.nn.Linear(last_size, len(self.expectiles) * output_size))
        self.convs = torch.nn.Conv1d(1, out_channels=n_kernels, kernel_size=1)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = hidden_sizes[-1] if output_size is None else output_size

    def forward(self, o):
        oprime = self.normL(o)
        stock, characteristics = o.split([self._conv_size, self._char_size], -1)
        stock_summary = self.convs(stock.view(-1, 1, self._conv_size)).mean(2)
        summarized_input = torch.cat((self.activ(stock_summary), characteristics), -1)
        res = self.model(summarized_input)
        return res.view(-1, self._output_size, len(self.expectiles))

    @property
    def output_size(self):
        return self._output_size


class DQN_Agent(object):
    def __init__(self, model_args, training_args, device, n_actions):
        self.device = device
        self.nets = DualNet(model_args, name="DQN", device=self.device)
        self.nets.target_net.load_state_dict(self.nets.policy_net.state_dict())
        for key in training_args:
            setattr(self, key, training_args[key])
        self.optimizer = optim.RMSprop(
            self.nets.policy_net.parameters(), lr=self.learning_rate
        )
        self.steps_done = 0
        self.memory = ReplayMemory(self.memory_size)
        self.n_actions = n_actions
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state, eval_mode=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold or eval_mode:
            with torch.inference_mode():
                return self.nets.policy_net(state).max(1)[1]
        else:
            return torch.tensor(
                random.choices(range(0, self.n_actions), k=state.shape[0]),
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = torch.diag(
            self.nets.policy_net(state_batch)[:, action_batch]
        )
        expected_state_action_values = (
            self.gamma * self.nets.target_net(next_state_batch).max(1)[1] + reward_batch
        )
        loss = self.criterion(state_action_values, expected_state_action_values)
        wandb.log({"batch loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class C51_Agent(object):
    def __init__(
        self,
        model_args,
        training_args,
        device,
        n_actions,
        vmin=0.0,
        vmax=2000.0,
        n_atoms=51,
        memory_size=100000,
    ):
        self.device = device
        self.nets = DualNet(model_args, name="C51", device=self.device)
        self.nets.target_net.load_state_dict(self.nets.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.nets.policy_net.parameters())
        for key in training_args:
            setattr(self, key, training_args[key])
        self.steps_done = 0

        self.memory = ReplayMemory(memory_size)
        self.n_actions = n_actions
        self.vmin = vmin
        self.vmax = vmax
        self.n_atoms = n_atoms
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.z_vector = torch.tensor([vmin + i * self.delta for i in range(n_atoms)])
        self.crit = nn.KLDivLoss()

    def select_action(self, state, eval_mode=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold or eval_mode:
            with torch.inference_mode():
                q_values = (self.nets.policy_net(state) * self.z_vector).sum(2)
                optimal_action = q_values.max(1)[1]
                return optimal_action
        else:
            return torch.tensor(
                random.choices(range(0, self.n_actions), k=state.shape[0]),
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        m = torch.zeros(self.n_atoms)
        q_values = (self.nets.policy_net(state_batch) * self.z_vector).sum(2)
        optimal_action = q_values.max(1)[1]
        out = self.nets.target_net(state_batch)
        optimal_action_dist = torch.stack(
            [out[i, optimal_action[i], :] for i in range(self.batch_size)]
        )
        actual_action = torch.stack(
            [out[i, action_batch[i], :] for i in range(self.batch_size)]
        )
        projection = (
            (self.gamma * self.z_vector.repeat(512, 1) + reward_batch.repeat(51, 1).t())
            .clamp(self.vmin, self.vmax)
            .double()
        )
        b = (projection - self.vmin) / self.delta
        l = torch.floor(b)
        u = torch.ceil(b)
        m = m + optimal_action_dist * (u - b)
        m = m + optimal_action_dist * (b - l)
        loss = self.crit(m / (actual_action + 0.0001), actual_action).neg()
        wandb.log({"batch loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QRDQN_Agent(object):
    def __init__(
        self, model_args, training_args, device, n_actions, memory_size=100000
    ):
        self.nets = DualNet(model_args, name="QRDQN", device=device)

        # self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=0.00001, max_lr=0.1, cycle_momentum=False)
        for key in training_args:
            setattr(self, key, training_args[key])
        self.steps_done = 0
        self.optimizer = optim.Adam(self.nets.policy_net.parameters(), lr=0.0005)
        self.memory = ReplayMemory(memory_size)
        self.n_actions = n_actions
        self.device = device
        self.quantiles = model_args["quantiles"]
        self.criterion = PinballHuber(quantiles=self.quantiles)

    def select_action(self, state, eval_mode=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold or eval_mode:
            with torch.inference_mode():
                return self.nets.policy_net(state).mean(2).max(1)[1]
        else:
            return torch.tensor(
                random.choices(range(0, self.n_actions), k=state.shape[0]),
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        quantile_values = self.nets.target_net(next_state_batch)
        optimal_action = quantile_values.mean(2).max(1)[1]
        optimal_action_dist = torch.stack(
            [quantile_values[i, optimal_action[i], :] for i in range(self.batch_size)]
        )
        obtained_quantiles = self.nets.policy_net(state_batch)
        obtained_dist = torch.stack(
            [obtained_quantiles[i, action_batch[i], :] for i in range(self.batch_size)]
        )
        optimal_state_action_values = (
            self.gamma * optimal_action_dist + reward_batch.view(-1, 1)
        )

        loss = self.criterion(obtained_dist, optimal_state_action_values)
        wandb.log({"batch loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ERDQN_Agent(object):
    def __init__(
        self, model_args, training_args, device, n_actions, memory_size=100000
    ):
        self.nets = DualNet(model_args, name="ERDQN", device=device)

        # self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=0.00001, max_lr=0.1, cycle_momentum=False)
        for key in training_args:
            setattr(self, key, training_args[key])
        self.steps_done = 0
        self.optimizer = optim.Adam(self.nets.policy_net.parameters(), lr=0.0005)
        self.memory = ReplayMemory(memory_size)
        self.n_actions = n_actions
        self.device = device
        self.expectiles = self.quantiles = model_args["quantiles"]
        self.criterion = ExpectileLoss(expectiles=self.expectiles)
        self.mean_index = len(self.expectiles) // 2 + 1

    def select_action(self, state, eval_mode=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold or eval_mode:
            with torch.inference_mode():
                return self.nets.policy_net(state)[:, :, self.mean_index].max(1)[1]
        else:
            torch.tensor(
                random.choices(range(0, self.n_actions), k=state.shape[0]),
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        expectile_values = self.nets.target_net(next_state_batch)
        optimal_action = expectile_values[:, :, self.mean_index].max(1)[1]
        optimal_action_dist = torch.stack(
            [expectile_values[i, optimal_action[i], :] for i in range(self.batch_size)]
        )
        obtained_expectiles = self.nets.policy_net(state_batch)
        obtained_dist = torch.stack(
            [obtained_expectiles[i, action_batch[i], :] for i in range(self.batch_size)]
        )
        optimal_state_action_values = (
            self.gamma * optimal_action_dist + reward_batch.view(-1, 1)
        )

        loss = self.criterion(obtained_dist, optimal_state_action_values)
        wandb.log({"batch loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class GTDQN_Agent(object):
    def __init__(
        self,
        model_args,
        training_args,
        device,
        n_actions,
    ):
        self.device = device
        self.nets = DualNet(model_args, name="ParameterDQN", device=self.device)
        for key in training_args:
            setattr(self, key, training_args[key])
        self.optimizer = optim.RMSprop(
            self.nets.policy_net.parameters(), lr=self.learning_rate
        )
        self.steps_done = 0
        self.memory = ReplayMemory(self.memory_size)
        self.n_actions = n_actions
        self.quantiles = model_args["quantiles"]
        self.criterion = PinballHuber(quantiles=self.quantiles, delta=1.0)

    def select_action(self, state, eval_mode=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold or eval_mode:
            with torch.inference_mode():
                lambdas = self.nets(state, "policy")
                means = (
                    lambdas[0]
                    + (1 / (1 + lambdas[1]) - 1 / (1 + lambdas[2])) / lambdas[3]
                )
                # means = torch.stack([gld_quantile(q, *lambdas) for q in self.quantiles],0).mean(0)
                return means.max(1)[1]
        else:
            return torch.tensor(
                random.choices(range(0, self.n_actions), k=state.shape[0]),
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        lambdas = [
            torch.diag(lambd[:, action_batch])
            for lambd in self.nets(state_batch, "policy")
        ]
        quantiles = [gld_quantile(q, *lambdas) for q in self.quantiles]

        state_action_values = torch.stack(quantiles, 1)
        lambdas = self.nets(next_state_batch, "target")
        means = lambdas[0] + (1 / (1 + lambdas[1]) - 1 / (1 + lambdas[2]) / lambdas[3])
        # means = torch.stack([gld_quantile(q, *lambdas) for q in self.quantiles],0).mean(0)
        optimal_action = means.max(1)[1]
        lambdas = [torch.diag(lambd[:, optimal_action]) for lambd in lambdas]
        optimal_quantiles = torch.stack(
            [gld_quantile(q, *lambdas) for q in self.quantiles], 1
        )
        expected_state_action_values = (
            self.gamma * optimal_quantiles + reward_batch.view(-1, 1)
        )
        loss = self.criterion(
            state_action_values, expected_state_action_values.squeeze()
        )
        wandb.log({"batch loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()


class sq_Agent(object):
    def __init__(self):
        print("SQ Policy created")

    def select_action(self, state, eval_mode=False):
        return

    def optimize(self):
        return
