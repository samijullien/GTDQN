from util import save__init__args
import torch
import torch.nn.functional as F


class LinearUtility:
    """The most basic utility function. It consists of a simple weighted linear combination of waste, sales and availability"""

    def __init__(
        self,
        alpha,
        beta,
        gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self,
        sales,
        waste,
        availability,
    ):
        return sales * self._alpha - waste * self._beta + availability * self._gamma


class CobbDouglasUtility:
    """Cobb-Douglas utility function."""

    def __init__(
        self,
        alpha,
        beta,
        gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self,
        sales,
        waste,
        availability,
    ):
        return (
            F.relu(sales) ** self._alpha
            * (1 + waste) ** -self._beta
            * availability**self._gamma
        )


class LogLinearUtility:
    """This utility is the linear combination of the logarithms of the values of interest."""

    def __init__(
        self,
        alpha,
        beta,
        gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self,
        sales,
        waste,
        availability,
    ):
        return (
            torch.log(1 + F.relu(sales)) * self._alpha
            - torch.log(torch.tensor(1.0 + waste)) * self._beta
            + torch.log(1.0 + availability) * self._gamma
        )


class HomogeneousReward:
    """Utility that returns an amount physically homogeneous to euros,
    but still takes availability into account
    """

    def __init__(
        self,
        alpha,
        beta,
        gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self,
        sales,
        waste,
        availability,
    ):
        return (availability**self._gamma * (sales - waste)).squeeze()


class CustomUtility:
    """Placeholder to create your own utility function"""

    def __init__(self, utility):
        self.txt = utility

    def reward(self, s, w, a):
        return eval(self.txt)
