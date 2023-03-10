import os

import numpy as np
import pandas as pd

import torch
import numpy
from util import Rscript

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)


class Assortment:
    def __init__(self, size, freshness=1, seed=None):
        """Create the assortment, a set of items. 
        Items are represented by vectors that 

        Args:
            size (_type_): _description_
            freshness (int, optional): _description_. Defaults to 1.
            seed (_type_, optional): _description_. Defaults to None.
        """
        self.size = size
        self.freshness = freshness
        self.seed = seed

        file_path = os.path.dirname(os.path.abspath(__file__))
        Rscript(size, seed, file_path)

        path_to_csv = os.path.join(file_path, "assortment.csv")
        df = pd.read_csv(path_to_csv).sample(n=size, random_state=seed)
        self.selling_price = torch.tensor(df.Price.to_numpy())
        self.cost = torch.tensor(df.Cost.to_numpy())

        # clamp base demand as some outliers in the data generation might ruin the purchase probability

        self.base_demand = (
            torch.tensor(df.Base_Demand.to_numpy())
            .clamp(0, 1000)
            .to(dtype=torch.float64)
        )
        self.shelf_lives = torch.round(
            torch.tensor(df.Shelf_life.to_numpy(), dtype=torch.float64) / freshness
        ).to(dtype=torch.float64)
        self.dims = torch.tensor(df.iloc[:, 1:4].values)
        self.characs = (
            torch.stack((self.selling_price, self.cost, self.shelf_lives))
            .t()
            .to(dtype=torch.float64)
        )

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "Cost": np.round(self.cost.numpy(), 2),
                "Price": np.round(self.selling_price.numpy(), 2),
                "Shelf life at purchase": self.shelf_lives.numpy(),
            }
        )

