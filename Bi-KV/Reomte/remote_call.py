import torch.distributed.rpc as rpc

def _call_remote_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)