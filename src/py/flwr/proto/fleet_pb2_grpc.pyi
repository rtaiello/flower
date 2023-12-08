"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import flwr.proto.fleet_pb2
import grpc

class FleetStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    CreateNode: grpc.UnaryUnaryMultiCallable[
        flwr.proto.fleet_pb2.CreateNodeRequest, flwr.proto.fleet_pb2.CreateNodeResponse
    ]

    DeleteNode: grpc.UnaryUnaryMultiCallable[
        flwr.proto.fleet_pb2.DeleteNodeRequest, flwr.proto.fleet_pb2.DeleteNodeResponse
    ]

    PullTaskIns: grpc.UnaryUnaryMultiCallable[
        flwr.proto.fleet_pb2.PullTaskInsRequest,
        flwr.proto.fleet_pb2.PullTaskInsResponse,
    ]
    """Retrieve one or more tasks, if possible

    HTTP API path: /api/v1/fleet/pull-task-ins
    """

    PushTaskRes: grpc.UnaryUnaryMultiCallable[
        flwr.proto.fleet_pb2.PushTaskResRequest,
        flwr.proto.fleet_pb2.PushTaskResResponse,
    ]
    """Complete one or more tasks, if possible

    HTTP API path: /api/v1/fleet/push-task-res
    """

class FleetServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def CreateNode(
        self,
        request: flwr.proto.fleet_pb2.CreateNodeRequest,
        context: grpc.ServicerContext,
    ) -> flwr.proto.fleet_pb2.CreateNodeResponse: ...
    @abc.abstractmethod
    def DeleteNode(
        self,
        request: flwr.proto.fleet_pb2.DeleteNodeRequest,
        context: grpc.ServicerContext,
    ) -> flwr.proto.fleet_pb2.DeleteNodeResponse: ...
    @abc.abstractmethod
    def PullTaskIns(
        self,
        request: flwr.proto.fleet_pb2.PullTaskInsRequest,
        context: grpc.ServicerContext,
    ) -> flwr.proto.fleet_pb2.PullTaskInsResponse:
        """Retrieve one or more tasks, if possible

        HTTP API path: /api/v1/fleet/pull-task-ins
        """
        pass
    @abc.abstractmethod
    def PushTaskRes(
        self,
        request: flwr.proto.fleet_pb2.PushTaskResRequest,
        context: grpc.ServicerContext,
    ) -> flwr.proto.fleet_pb2.PushTaskResResponse:
        """Complete one or more tasks, if possible

        HTTP API path: /api/v1/fleet/push-task-res
        """
        pass

def add_FleetServicer_to_server(
    servicer: FleetServicer, server: grpc.Server
) -> None: ...
