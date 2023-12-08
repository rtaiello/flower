import argparse
import warnings
from collections import OrderedDict

import torch
from flamingo import Flamingo
from flwr_datasets import FederatedDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from model import Net
from flwr.common.secure_aggregation.quantization import quantize, multiply

import flwr as fl
from flwr.client.secure_aggregation import SecAggPlusHandler

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data(node_id):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(node_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node-id",
    choices=[0, 1],
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
my_node_id = parser.parse_args().node_id
node_ids = [0, 1]

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
num_params = torch.tensor([p.numel() for p in net.parameters()]).sum()
trainloader, testloader = load_data(node_id=my_node_id)


# Define Flamingo client
class FlamingoClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self._flamingo = Flamingo()
        self._flamingo.setup_pairwise_secrets(
            my_node_id=my_node_id, nodes_ids=node_ids, num_params=num_params + 1
        )

    def get_parameters(self, config):
        params = parameters_to_vector(net.parameters()).detach().cpu().numpy()
        curr_round = config["curr_round"]
        num_samples = len(trainloader.dataset)
        quantized_params = quantize(params)
        quantized_params = multiply(quantized_params, num_samples)
        quantized_params = [num_samples] + quantized_params
        encrypted_params = self._flamingo.protect(
            curr_round, quantized_params, node_ids
        )
        return [encrypted_params]

    def set_parameters(self, parameters):
        parameters = parameters[0]
        vector_to_parameters(torch.tensor(parameters), net.parameters())

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config=config), 0, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlamingoClient(),
)
