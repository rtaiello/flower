from typing import List, Tuple

import torch
from model import Net

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters


def get_initial_parameters():
    net = Net()
    params = (
        torch.nn.utils.parameters_to_vector(net.parameters()).detach().cpu().numpy()
    )
    return ndarrays_to_parameters([params])


def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"curr_round": server_round}
        return fit_config

    return fit_config_fn


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FlamingoFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=get_on_fit_config(),
    initial_parameters=get_initial_parameters(),
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
