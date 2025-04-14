from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Signal(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIGNAL_SEND: _ClassVar[Signal]
    SIGNAL_RECV: _ClassVar[Signal]
    SIGNAL_ACK: _ClassVar[Signal]
    SIGNAL_CHECK: _ClassVar[Signal]
    SIGNAL_SKIP: _ClassVar[Signal]
    SIGNAL_TERMINATE: _ClassVar[Signal]
SIGNAL_SEND: Signal
SIGNAL_RECV: Signal
SIGNAL_ACK: Signal
SIGNAL_CHECK: Signal
SIGNAL_SKIP: Signal
SIGNAL_TERMINATE: Signal

class IdTokenPair(_message.Message):
    __slots__ = ("id", "token_num")
    ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_NUM_FIELD_NUMBER: _ClassVar[int]
    id: int
    token_num: int
    def __init__(self, id: _Optional[int] = ..., token_num: _Optional[int] = ...) -> None: ...

class TaskInfo(_message.Message):
    __slots__ = ("request_id", "id", "infer_worker", "cache_worker", "token_num", "index", "task_type", "type", "task_num", "weight", "cache_pages_list")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    INFER_WORKER_FIELD_NUMBER: _ClassVar[int]
    CACHE_WORKER_FIELD_NUMBER: _ClassVar[int]
    TOKEN_NUM_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TASK_NUM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    CACHE_PAGES_LIST_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    id: int
    infer_worker: int
    cache_worker: int
    token_num: int
    index: int
    task_type: int
    type: str
    task_num: int
    weight: int
    cache_pages_list: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, request_id: _Optional[int] = ..., id: _Optional[int] = ..., infer_worker: _Optional[int] = ..., cache_worker: _Optional[int] = ..., token_num: _Optional[int] = ..., index: _Optional[int] = ..., task_type: _Optional[int] = ..., type: _Optional[str] = ..., task_num: _Optional[int] = ..., weight: _Optional[int] = ..., cache_pages_list: _Optional[_Iterable[int]] = ...) -> None: ...

class CombindedTaskInfo(_message.Message):
    __slots__ = ("request_id", "id", "infer_worker", "cache_worker", "token_num", "index", "task_type", "type", "task_num", "cache_pages_list", "id_token_pair")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    INFER_WORKER_FIELD_NUMBER: _ClassVar[int]
    CACHE_WORKER_FIELD_NUMBER: _ClassVar[int]
    TOKEN_NUM_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TASK_NUM_FIELD_NUMBER: _ClassVar[int]
    CACHE_PAGES_LIST_FIELD_NUMBER: _ClassVar[int]
    ID_TOKEN_PAIR_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    id: int
    infer_worker: int
    cache_worker: int
    token_num: int
    index: int
    task_type: int
    type: str
    task_num: int
    cache_pages_list: _containers.RepeatedCompositeFieldContainer[PageList]
    id_token_pair: _containers.RepeatedCompositeFieldContainer[IdTokenPair]
    def __init__(self, request_id: _Optional[int] = ..., id: _Optional[int] = ..., infer_worker: _Optional[int] = ..., cache_worker: _Optional[int] = ..., token_num: _Optional[int] = ..., index: _Optional[int] = ..., task_type: _Optional[int] = ..., type: _Optional[str] = ..., task_num: _Optional[int] = ..., cache_pages_list: _Optional[_Iterable[_Union[PageList, _Mapping]]] = ..., id_token_pair: _Optional[_Iterable[_Union[IdTokenPair, _Mapping]]] = ...) -> None: ...

class PageList(_message.Message):
    __slots__ = ("cache_pages_list",)
    CACHE_PAGES_LIST_FIELD_NUMBER: _ClassVar[int]
    cache_pages_list: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, cache_pages_list: _Optional[_Iterable[int]] = ...) -> None: ...

class TaskInfoList(_message.Message):
    __slots__ = ("tasks",)
    TASKS_FIELD_NUMBER: _ClassVar[int]
    tasks: _containers.RepeatedCompositeFieldContainer[TaskInfo]
    def __init__(self, tasks: _Optional[_Iterable[_Union[TaskInfo, _Mapping]]] = ...) -> None: ...

class ComfirmationMessage(_message.Message):
    __slots__ = ("msg",)
    MSG_FIELD_NUMBER: _ClassVar[int]
    msg: str
    def __init__(self, msg: _Optional[str] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartRequest(_message.Message):
    __slots__ = ("msg",)
    MSG_FIELD_NUMBER: _ClassVar[int]
    msg: str
    def __init__(self, msg: _Optional[str] = ...) -> None: ...
