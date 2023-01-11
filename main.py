import toml
import sys
import argparse
import torch
import wandb
import env
import code_base
import os
from tqdm import tqdm
import datetime

agents = __import__("code_base")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default='DQN_Agent',
        help="Agent to use",
    )
    parser.add_argument(
        "--forecast-variance",
        type=float,
        default=0.1,
        help="variance of forecast receive by the agent",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=1,
        help="Number of quantiles/expectiles to consider",
    )

    parser.add_argument
    args = parser.parse_args()
    # fmt: on
    return args


def train(model, run_name, env_args, num_episodes):
    for i_episode in tqdm(range(num_episodes)):
        e = env.StoreEnv(**env_args, seed=i_episode)
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
        torch.save(model.nets.policy_net.state_dict(), run_name + ".pt")
    print("Complete")


def evaluate(model, run_name, args_env, n_env, steps):
    with torch.inference_mode():
        overall_sales = []
        overall_waste = []
        for env_id in tqdm(range(n_env)):
            e = env.StoreEnv(**env_args, seed=1000 + env_id)
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
        torch.save(torch.cat(overall_sales).cpu(), "res/sales" + run_name + ".pt")
        torch.save(torch.cat(overall_waste).cpu(), "res/waste" + run_name + ".pt")


def evaluate_baseline(env_args, n_env, steps):
    with torch.inference_mode():
        overall_sales = []
        overall_waste = []
        for env_id in tqdm(range(n_env)):
            e = env.StoreEnv(**env_args, seed=1000 + env_id)
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
        run_name = args.agent_name+str(args.forecast_variance)
        torch.save(torch.cat(overall_sales).cpu(), "res/BLsales" + run_name + ".pt")
        torch.save(torch.cat(overall_waste).cpu(), "res/BLwaste" + run_name + ".pt")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    config = toml.load('config.toml')
    args = parse_args()

    training_args = config["training_args"]
    args_model = config["args_model"]
    env_args = config["args_env"]
    env_args['forecast_variance'] = args.forecast_variance
    e = env.StoreEnv(seed=0)
    n_actions = e._action_space.n
    wandb.config = config    
    print(torch.cuda.device_count())
    quantiles = torch.arange(0.0,1,1/args.n_quantiles)[1:]
    args_model['quantiles'] = torch.arange(0.0,1,1/args.n_quantiles)[1:]
    run_name = args.agent_name+str(args.forecast_variance)
    wandb.init(
        config=config,
        name=run_name,
        project=config["global"]["proj_name"],
        tags=config["global"]["tags"],
    )
    agent = getattr(agents, args.agent_name)
    model = agent(args_model, training_args, device, n_actions)
    train(model, run_name, env_args, 60)
    evaluate(model, run_name, env_args, 30, 2000)
    evaluate_baseline(env_args, 30, 2000)

