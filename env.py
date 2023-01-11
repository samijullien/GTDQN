import gym
from gym import spaces
import torch
import torch.distributions as d
import torch.nn.functional as F
import numpy as np
from inspect import getfullargspec
from math import pi

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.cuda.set_device("cuda:0")

from assortment import Assortment


def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    prefix = "_" if underscore else ""
    self = values["self"]
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if "__init__" in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])


class LinearUtility:
    def __init__(
        self, alpha, beta, gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self, sales, waste, availability,
    ):
        return sales * self._alpha - waste * self._beta + availability * self._gamma


class StoreEnv(gym.Env):
    def __init__(
        self,
        assortment_size=100,  # number of items to train
        max_stock=250,  # Size of maximum stock
        horizon=2500,
        substep_count=2,
        bucket_customers=[500.0, 900.0],
        covariance_buckets=[1.0, 0.0, 0.0, 1.0],
        forecast_bias=0.0,
        forecast_variance=0.0001,
        freshness=2,
        utility_function="linear",
        utility_weights={"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        characDim=4,
        forecast_horizon=7,  # How many days ahead do we haev a forecast?
        lead_time=1,  # Defines how quickly the orders goes through the buffer - also impacts the relevance of the observation
        lead_time_fast=0,  # To have a different lead time at end of day
        seed=None,
    ):
        save__init__args(locals(), underscore=True)
        self.bucket_customers = torch.tensor(bucket_customers, dtype=torch.float64)
        self.bucket_cov = torch.tensor(covariance_buckets, dtype=torch.float64).view(
            len(bucket_customers), len(bucket_customers)
        )

        # Spaces

        self._action_space = spaces.Discrete(max_stock)
        # Consider the action space to be discrete for DQN purposes
        self.stock = torch.zeros(assortment_size, max_stock, requires_grad=False)

        # correct high with max shelf life

        self._horizon = int(horizon)
        self.assortment = Assortment(assortment_size, freshness, seed)
        self._repeater = (
            torch.stack(
                (self.assortment.shelf_lives, torch.zeros(self._assortment_size))
            )
            .transpose(0, 1)
            .reshape(-1)
            .detach()
        )
        self.forecast = torch.zeros(assortment_size, 1)
        self._step_counter = 0
        self.sales = 0
        self.total_waste = torch.zeros(self.assortment.size)
        self.availability = 0.0
        self.total_availability = 0.0
        self.sales = 0
        self.total_sales = 0.0

        self._customers = d.multivariate_normal.MultivariateNormal(
            self.bucket_customers, self.bucket_cov
        )
        self.assortment.base_demand = (
            self.assortment.base_demand.detach() / self.bucket_customers.sum()
        )
        self._bias = d.normal.Normal(forecast_bias, forecast_variance)

        self.expected_mean = self.bucket_customers.sum().view(-1, 1) * (
            self.assortment.base_demand
            * (self.assortment.selling_price - self.assortment.cost)
        )
        self.expected_mean = self.expected_mean.squeeze()
        self.expected_std = (
            self.bucket_customers.sum().view(-1, 1)
            * (
                self.assortment.base_demand
                * (self.assortment.selling_price - self.assortment.cost)
            )
            * (
                torch.ones_like(self.assortment.base_demand)
                - self.assortment.base_demand
            )
        )
        self.expected_std = self.expected_std.squeeze()
        # We want a yearly seasonality - We have a cosinus argument and a phase.
        # Note that, as we take the absolute value, 2*pi/365 becomes pi/365.

        self._year_multiplier = torch.arange(0.0, horizon + 100, pi / 365)
        self._week_multiplier = torch.arange(0.0, horizon + 100, pi / 7)
        self._phase = 2 * pi * torch.rand(assortment_size)
        self._phase2 = 2 * pi * torch.rand(assortment_size)
        self.forecast_horizon = forecast_horizon * substep_count
        loc = forecast_bias * torch.ones(self.forecast_horizon)
        scale = torch.diag(
            torch.arange(1, 1 + forecast_horizon).repeat_interleave(2)
            * forecast_variance
        )

        self.forecast_drift = d.multivariate_normal.MultivariateNormal(loc, scale)
        self.create_buffers(lead_time, lead_time_fast)
        self._observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(
                assortment_size,
                max_stock + characDim + lead_time + lead_time_fast + forecast_horizon,
            ),
        )
        if utility_function == "linear":
            self.utility_function = LinearUtility(**utility_weights)
            # elif utility_function == "loglinear":
            #    self.utility_function = LogLinearUtility(**utility_weights)
            # elif utility_function == "cobbdouglas":
            #    self.utility_function = CobbDouglasUtility(**utility_weights)
            # elif utility_function == "homogeneous":
            #    self.utility_function = HomogeneousReward(**utility_weights)
        else:
            self.utility_function = utility_function
        self._updateEnv()
        for i in range(self._lead_time):
            units_to_order = (
                torch.as_tensor(self.forecast.squeeze() * self.bucket_customers[i])
                .round()
                .clamp(0, self._max_stock)
            )
            self._addStock(units_to_order)

    def reset(self):
        self._updateObs()
        self._step_counter = 0
        return self.get_obs()

    @torch.inference_mode()
    def step(self, action):
        new_action = (
            torch.as_tensor(action, dtype=torch.int32)
            .clamp(0, self._max_stock)
            .to("cuda:0")
        )
        if self.day_position % self._substep_count == 0:
            order_cost = self._make_fast_order(new_action)
            (sales, availability) = self._generateDemand(self.real.clamp_(0.0, 1.0))
            waste = self._waste()  # Update waste and store result
            self._reduceShelfLives()
            self._step_counter += 1
            self._updateEnv()
        else:

            self.day_position += 1
            order_cost = self._make_order(new_action)
            (sales, availability) = self._generateDemand(self.real.clamp_(0.0, 1.0))
            waste = torch.zeros(
                self._assortment_size
            )  # By default, no waste before the end of day
            self._updateObs()
        # sales.sub_(order_cost)
        self.sales = sales
        self.total_sales += sales
        self.waste = waste
        self.total_waste += waste
        self.availability = availability

        utility = self.utility_function.reward(sales, waste, availability)
        done = self._step_counter == self.horizon
        return (self.get_obs(), utility, done, (sales, waste))

    def get_obs(self):
        return self._obs

    def run_to_completion(self, order, n_customers):
        done = False
        obs = self.reset()
        rewards = []
        while not done:
            customers = self.bucket_customers.mean().round()
            stock = self.get_full_inventory_position()
            forecast = self.forecast.squeeze()
            std = torch.sqrt(customers * forecast + (1 - forecast))
            number = F.relu(eval(order)).round()
            # Step the environment and get its observation
            obs = self.step(number.numpy())
            # Store reward for the specific time step
            rewards.append(obs[1].sum())
            done = obs[2]
        return rewards

    def render(self, mode="human", close=False):
        print(f"Step: {self._step_counter}")
        print(f"Step sales: {self.sales},Total sales: {self.total_sales})")
        print(f"Step Waste: {self.waste}, (Total waste: {self.total_waste})")
        print(
            f"Availability: {self.availability}, (Mean Availability: {self.total_availability/self._step_counter})"
        )

    # ##########################################################################
    # Helpers
    @torch.inference_mode()
    def _updateObs(self):
        self._obs = torch.cat(
            (
                self.stock,
                self.assortment.characs,
                self.seen_forecast,
                torch.stack(self._buffer + self._buffer_fast, 1),
                torch.ones(self._assortment_size, 1) * self.day_position,
            ),
            1,
        )

    @torch.inference_mode()
    def _updateEnv(self):
        self.day_position = 1
        arguments = (
            self._year_multiplier[
                self._step_counter : self._step_counter + self.forecast_horizon
            ].view(-1, 1)
            + self._phase
        )
        arguments2 = (
            self._week_multiplier[
                self._step_counter : self._step_counter + self.forecast_horizon
            ].view(-1, 1)
            + self._phase2
        )
        self.seen_forecast = (
            self.assortment.base_demand * arguments.cos().abs() * arguments2.cos().abs()
        ).t() + self.forecast_drift.sample((self._assortment_size,))
        self.real = self.seen_forecast[:, 0].view(-1) * (
            1 + self._bias.sample((self._assortment_size,))
        )
        self._updateObs()

    @torch.inference_mode()
    def _addStock(self, units):
        padding = self._max_stock - units
        replenishment = torch.stack((units, padding)).t().reshape(-1)
        restock_matrix = self._repeater.repeat_interleave(
            repeats=replenishment.long(), dim=0
        ).view(self._assortment_size, self._max_stock)
        torch.add(
            self.stock.sort(1)[0],
            restock_matrix.sort(1, descending=True)[0],
            out=self.stock,
        )
        # total_units = restock_matrix.ge(1).sum(1).add_(self.stock.ge(1).sum(1))
        # penalty_cost_forbidden = (
        #    F.relu(total_units - self._max_stock)
        #    .double()
        #    .mul_(self.assortment.selling_price)
        # )
        # return penalty_cost_forbidden
        return

    @torch.inference_mode()
    def _sellUnits(self, units):
        sold = torch.min(self.stock.ge(1).sum(1).double(), units)
        availability = self.stock.ge(1).sum(1).double().div(units).clamp(0, 1)
        availability[torch.isnan(availability)] = 1.0
        reward = (
            sold.mul_(2)
            .sub_(units)
            .mul(self.assortment.selling_price - self.assortment.cost)
        )
        (p, n) = self.stock.shape
        stock_vector = self.stock.sort(1, descending=True)[0].view(-1)
        to_keep = n - units

        interleaver = torch.stack((units, to_keep)).t().reshape(2, p).view(-1).long()
        binary_vec = torch.tensor([0.0, 1]).repeat(p).repeat_interleave(interleaver)
        self.stock = binary_vec.mul_(stock_vector).view(p, n)
        return (reward, availability)

    @torch.inference_mode()
    def _waste(self):
        waste = torch.mul(
            self.stock.eq(1).sum(1).double(), self.assortment.selling_price
        )
        return waste

    @torch.inference_mode()
    def _reduceShelfLives(self):
        self.stock = F.relu(self.stock - 1)

    @torch.inference_mode()
    def _generateDemand(self, consumption_prob):
        sampled_customers = (
            self._customers.sample().round().int()[self.day_position - 1]
        )
        purchases_gen = d.bernoulli.Bernoulli(consumption_prob)
        demand = (
            purchases_gen.sample((sampled_customers,)).sum(0).clamp(0, self._max_stock)
        )
        (reward, availability) = self._sellUnits(demand)
        return (reward, availability)

    # Updates stock matrix and transportation cost (reward)
    # order speed increases the speed of all orders currently in the buffer.

    def _make_order(self, units):
        self._buffer.append(units.double().view(-1))
        penaltyCost = self._addStock(self._buffer.pop(0))
        return penaltyCost

    def _make_fast_order(self, units):
        self._buffer_fast.append(units.double().view(-1))
        penaltyCost = self._addStock(self._buffer_fast.pop(0))
        return penaltyCost

    def get_partial_position(self):
        return self.stock.ge(1).sum(1).double()

    def get_full_inventory_position(self):
        ip = self.get_partial_position()
        ip += torch.stack(self._buffer).sum(0)
        return ip

    def create_buffers(self, slow_speed, fast_speed):
        self._buffer = []
        self._buffer_fast = []
        for i in range(slow_speed):
            self._buffer.append(torch.zeros(self._assortment_size))
        for i in range(fast_speed):
            self._buffer_fast.append(torch.zeros(self._assortment_size))

    def transportation_cost(
        self, units, transport_size=300000, transport_cost=250.0,
    ):
        volume = units * self.assortment.dims.t().sum(0)
        number_of_trucks = np.trunc(volume.sum() / transport_size) + 1

        # This +1 has no impact even if the order is 0 as we return the contribution to the total cost, not the total cost itself

        total_cost = number_of_trucks * transport_cost
        return volume / volume.sum() * total_cost

    # ##########################################################################
    # Properties

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def horizon(self):
        return self._horizon
