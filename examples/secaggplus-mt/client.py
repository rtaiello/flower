import time

import numpy as np

import flwr as fl
from flwr.client.secure_aggregation import SecAggPlusHandler
from flwr.common import Code, FitIns, FitRes, Status
from flwr.common.parameter import ndarrays_to_parameters


# Define Flower client with the SecAgg+ protocol
class FlowerClient(fl.client.Client, SecAggPlusHandler):
    def fit(self, fit_ins: FitIns) -> FitRes:
        ret_vec = [np.ones(3)]
        ret = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(ret_vec),
            num_examples=1,
            metrics={},
        )
        # Force a significant delay for testing purposes
        if self._shared_state.sid == 0:
            print(f"Client {self._shared_state.sid} dropped for testing purposes.")
            time.sleep(4)
            return ret
        print(f"Client {self._shared_state.sid} uploading {ret_vec[0]}...")
        return ret


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:9092",
    client=FlowerClient(),
    transport="grpc-rere",
)
