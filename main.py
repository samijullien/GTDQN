import yaml
import sys

import torch
import wandb
import env
import code_base
from tqdm import tqdm
import datetime

agents = __import__("code_base")

config_file = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
with open(config_file, "r") as f:
    config = yaml.load(f, yaml.SafeLoader)

training_args = config["training_args"]
kwargsModel = config["kwargsModel"]
Agent_Name = config["Agent_Name"]
kwargsEnv = config["kwargsEnv"]
e = env.StoreEnv(seed=0)
n_actions = e._action_space.n
wandb.config = config


def train(model, kwargsEnv, num_episodes):
    for i_episode in tqdm(range(num_episodes)):
        e = env.StoreEnv(**kwargsEnv, seed=i_episode)
        done = False
        state = e.get_obs()
        while not done:
            action = model.select_action(state).clamp(
                e._max_stock - e.stock.count_nonzero(1)
            )
            next_state, reward, done, (sales, waste) = e.step(action)
            wandb.log(
                {
                    "Reward": reward.sum().item(),
                    "Sales": sales.sum().item(),
                    "Waste": e.total_waste.sum(),
                }
            )
            standardized_reward = (reward - e.expected_mean) / e.expected_std
            model.memory.push(
                state.split(1), action.split(1), next_state.split(1), reward.split(1),
            )
            state = next_state
            model.optimize_model()
            if done:
                break

        if i_episode % model.target_update == 0:
            model.nets.target_net.load_state_dict(model.nets.policy_net.state_dict())
            model.nets.target_net.eval()
        torch.save(model.nets.policy_net.state_dict(), config["global"]["name"] + ".pt")
    print("Complete")


def evaluate(model, kwargsEnv, n_env, steps):
    with torch.inference_mode():
        overall_sales = []
        overall_waste = []
        for env_id in tqdm(range(n_env)):
            e = env.StoreEnv(**kwargsEnv, seed=1000 + env_id)
            state = e.get_obs()
            rewards = []
            for i in range(steps):
                action = model.select_action(state, eval_mode=True).clamp(
                    e._max_stock - e.stock.count_nonzero(1)
                )
                state, reward, done, (sales, waste) = e.step(action)
                overall_waste.append(waste)
                overall_sales.append(sales)
                rewards.append(reward.mean().item())
            res = sum(rewards) / len(rewards)
            print(res)
            wandb.log({"Average reward ": res})
        run_name = config["global"]["name"]
        torch.save(torch.cat(overall_sales).cpu(), "res/sales" + run_name + ".pt")
        torch.save(torch.cat(overall_waste).cpu(), "res/waste" + run_name + ".pt")


def evaluate_baseline(kwargsEnv, n_env, steps):
    with torch.inference_mode():
        overall_sales = []
        overall_waste = []
        for env_id in tqdm(range(n_env)):
            e = env.StoreEnv(**kwargsEnv, seed=1000 + env_id)
            state = e.get_obs()
            rewards = []
            for i in range(steps):
                mask = (e.stock.count_nonzero(1) < e.assortment.base_demand).type(
                    torch.uint8
                )
                action = mask * (
                    2 * e.assortment.base_demand * e.bucket_customers.sum()
                ).clamp(e._max_stock - e.stock.count_nonzero(1))
                state, reward, done, (sales, waste) = e.step(action)
                overall_waste.append(waste)
                overall_sales.append(sales)
                rewards.append(reward.mean().item())
            res = sum(rewards) / len(rewards)
            print(res)
            wandb.log({"Average reward ": res})
        run_name = config["global"]["name"]
        torch.save(torch.cat(overall_sales).cpu(), "res/BLsales" + run_name + ".pt")
        torch.save(torch.cat(overall_waste).cpu(), "res/BLwaste" + run_name + ".pt")


if __name__ == "__main__":
    print(device)
    print(torch.cuda.device_count())
    wandb.init(
        config=config,
        name=config["global"]["name"],
        project=config["global"]["proj_name"],
        tags=config["global"]["tags"],
    )
    agent = getattr(agents, Agent_Name)
    model = agent(kwargsModel, training_args, device, n_actions)
    train(model, kwargsEnv, 60)
    evaluate(model, kwargsEnv, 30, 2000)
    # evaluate_baseline(kwargsEnv, 30, 2000)

