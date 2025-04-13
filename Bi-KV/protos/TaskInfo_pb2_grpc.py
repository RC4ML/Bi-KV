# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from protos import TaskInfo_pb2 as TaskInfo__pb2

GRPC_GENERATED_VERSION = '1.71.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in TaskInfo_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class InferWorkerServiceStub(object):
    """InferWorker 服务
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReceiveTasksFromScheduler = channel.unary_unary(
                '/InferWorkerService/ReceiveTasksFromScheduler',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.SendKVCacheData = channel.unary_unary(
                '/InferWorkerService/SendKVCacheData',
                request_serializer=TaskInfo__pb2.CombindedTaskInfo.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.RecvKVCacheData = channel.unary_unary(
                '/InferWorkerService/RecvKVCacheData',
                request_serializer=TaskInfo__pb2.CombindedTaskInfo.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.StartWriteCacheData = channel.unary_unary(
                '/InferWorkerService/StartWriteCacheData',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.ShutDown = channel.unary_unary(
                '/InferWorkerService/ShutDown',
                request_serializer=TaskInfo__pb2.Empty.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)


class InferWorkerServiceServicer(object):
    """InferWorker 服务
    """

    def ReceiveTasksFromScheduler(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendKVCacheData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RecvKVCacheData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartWriteCacheData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ShutDown(self, request, context):
        """关闭 infer worker
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferWorkerServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReceiveTasksFromScheduler': grpc.unary_unary_rpc_method_handler(
                    servicer.ReceiveTasksFromScheduler,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'SendKVCacheData': grpc.unary_unary_rpc_method_handler(
                    servicer.SendKVCacheData,
                    request_deserializer=TaskInfo__pb2.CombindedTaskInfo.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'RecvKVCacheData': grpc.unary_unary_rpc_method_handler(
                    servicer.RecvKVCacheData,
                    request_deserializer=TaskInfo__pb2.CombindedTaskInfo.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'StartWriteCacheData': grpc.unary_unary_rpc_method_handler(
                    servicer.StartWriteCacheData,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'ShutDown': grpc.unary_unary_rpc_method_handler(
                    servicer.ShutDown,
                    request_deserializer=TaskInfo__pb2.Empty.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'InferWorkerService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('InferWorkerService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class InferWorkerService(object):
    """InferWorker 服务
    """

    @staticmethod
    def ReceiveTasksFromScheduler(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferWorkerService/ReceiveTasksFromScheduler',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SendKVCacheData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferWorkerService/SendKVCacheData',
            TaskInfo__pb2.CombindedTaskInfo.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RecvKVCacheData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferWorkerService/RecvKVCacheData',
            TaskInfo__pb2.CombindedTaskInfo.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def StartWriteCacheData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferWorkerService/StartWriteCacheData',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ShutDown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/InferWorkerService/ShutDown',
            TaskInfo__pb2.Empty.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class CacheCoordinatorServiceStub(object):
    """CacheCoordinator 服务
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReceiveTasksFromInferWorker = channel.unary_unary(
                '/CacheCoordinatorService/ReceiveTasksFromInferWorker',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.ReceiveTasksFromScheduler = channel.unary_unary(
                '/CacheCoordinatorService/ReceiveTasksFromScheduler',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.ComfirmationMessage.FromString,
                _registered_method=True)
        self.PollBatchFromInferWorker = channel.unary_unary(
                '/CacheCoordinatorService/PollBatchFromInferWorker',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.ComfirmationMessage.FromString,
                _registered_method=True)
        self.StartProcessRequest = channel.unary_unary(
                '/CacheCoordinatorService/StartProcessRequest',
                request_serializer=TaskInfo__pb2.StartRequest.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)
        self.ShutDown = channel.unary_unary(
                '/CacheCoordinatorService/ShutDown',
                request_serializer=TaskInfo__pb2.Empty.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)


class CacheCoordinatorServiceServicer(object):
    """CacheCoordinator 服务
    """

    def ReceiveTasksFromInferWorker(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReceiveTasksFromScheduler(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PollBatchFromInferWorker(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartProcessRequest(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ShutDown(self, request, context):
        """关闭 cache coordinator
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CacheCoordinatorServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReceiveTasksFromInferWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.ReceiveTasksFromInferWorker,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'ReceiveTasksFromScheduler': grpc.unary_unary_rpc_method_handler(
                    servicer.ReceiveTasksFromScheduler,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.ComfirmationMessage.SerializeToString,
            ),
            'PollBatchFromInferWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.PollBatchFromInferWorker,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.ComfirmationMessage.SerializeToString,
            ),
            'StartProcessRequest': grpc.unary_unary_rpc_method_handler(
                    servicer.StartProcessRequest,
                    request_deserializer=TaskInfo__pb2.StartRequest.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
            'ShutDown': grpc.unary_unary_rpc_method_handler(
                    servicer.ShutDown,
                    request_deserializer=TaskInfo__pb2.Empty.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'CacheCoordinatorService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('CacheCoordinatorService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class CacheCoordinatorService(object):
    """CacheCoordinator 服务
    """

    @staticmethod
    def ReceiveTasksFromInferWorker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/CacheCoordinatorService/ReceiveTasksFromInferWorker',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ReceiveTasksFromScheduler(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/CacheCoordinatorService/ReceiveTasksFromScheduler',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.ComfirmationMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def PollBatchFromInferWorker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/CacheCoordinatorService/PollBatchFromInferWorker',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.ComfirmationMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def StartProcessRequest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/CacheCoordinatorService/StartProcessRequest',
            TaskInfo__pb2.StartRequest.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ShutDown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/CacheCoordinatorService/ShutDown',
            TaskInfo__pb2.Empty.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class KVCacheServiceStub(object):
    """KVCache 服务
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReceiveTasksFromCoordinator = channel.unary_unary(
                '/KVCacheService/ReceiveTasksFromCoordinator',
                request_serializer=TaskInfo__pb2.TaskInfoList.SerializeToString,
                response_deserializer=TaskInfo__pb2.ComfirmationMessage.FromString,
                _registered_method=True)
        self.ShutDown = channel.unary_unary(
                '/KVCacheService/ShutDown',
                request_serializer=TaskInfo__pb2.Empty.SerializeToString,
                response_deserializer=TaskInfo__pb2.Empty.FromString,
                _registered_method=True)


class KVCacheServiceServicer(object):
    """KVCache 服务
    """

    def ReceiveTasksFromCoordinator(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ShutDown(self, request, context):
        """关闭 KVCache
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_KVCacheServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReceiveTasksFromCoordinator': grpc.unary_unary_rpc_method_handler(
                    servicer.ReceiveTasksFromCoordinator,
                    request_deserializer=TaskInfo__pb2.TaskInfoList.FromString,
                    response_serializer=TaskInfo__pb2.ComfirmationMessage.SerializeToString,
            ),
            'ShutDown': grpc.unary_unary_rpc_method_handler(
                    servicer.ShutDown,
                    request_deserializer=TaskInfo__pb2.Empty.FromString,
                    response_serializer=TaskInfo__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'KVCacheService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('KVCacheService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class KVCacheService(object):
    """KVCache 服务
    """

    @staticmethod
    def ReceiveTasksFromCoordinator(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/KVCacheService/ReceiveTasksFromCoordinator',
            TaskInfo__pb2.TaskInfoList.SerializeToString,
            TaskInfo__pb2.ComfirmationMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ShutDown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/KVCacheService/ShutDown',
            TaskInfo__pb2.Empty.SerializeToString,
            TaskInfo__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
