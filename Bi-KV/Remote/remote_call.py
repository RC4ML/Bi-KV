import torch.distributed.rpc as rpc

def call_remote_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)