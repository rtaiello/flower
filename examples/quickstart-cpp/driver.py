from fedavg_cpp import FedAvgCpp

import flwr as fl

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.driver.start_driver(
        server_address="0.0.0.0:9091",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=FedAvgCpp(),
    )
