"""The one wire between a simulator server and its host, for every benchmark: an 8-byte length, a JSON header, then the
raw bytes of the arrays the header names, on 127.0.0.1 with one client at a time.

    sim = Sim(port)                          # the host's end
    rep = sim.request(op="reset", seed=1)    # a reply carrying "error" raises RuntimeError; so does a lost connection

Besides JSON values and numpy arrays it carries `roboskills.eval.interface` dataclasses, and callables by reference:
module-level functions of `roboskills` modules and `functools.partial`s of them (a `Hand.model_q`, say). Stdlib and
numpy only, so simulator images import it without the host's packages.
"""

import dataclasses
import functools
import importlib
import inspect
import json
import socket
from pathlib import PurePath

import numpy as np


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ref(f):
    """'module:qualname' of a module-level `roboskills` function, which the other end imports (not a bound method:
    its object would not cross)."""
    name = f"{getattr(f, '__module__', '')}:{getattr(f, '__qualname__', '')}"
    if not name.startswith("roboskills.") or "<" in name or inspect.ismethod(f):
        raise TypeError(f"{f!r} cannot cross the wire: use a module-level roboskills function or a partial of one")
    return name


def send(sock, obj):
    arrays = []

    def enc(x):
        if isinstance(x, np.ndarray):
            arrays.append(np.ascontiguousarray(x))
            return {"__nd__": [x.dtype.str, list(x.shape)]}
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return {"__dc__": type(x).__name__, **{f.name: enc(getattr(x, f.name)) for f in dataclasses.fields(x)}}
        if isinstance(x, functools.partial):
            return {"__fn__": ref(x.func), "args": enc(list(x.args)), "kw": enc(x.keywords)}
        if callable(x):
            return {"__fn__": ref(x)}
        if isinstance(x, dict):
            return {str(k): enc(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [enc(v) for v in x]
        if isinstance(x, PurePath):
            return str(x)
        return x.item() if isinstance(x, np.generic) else x

    head = json.dumps(enc(obj)).encode()
    sock.sendall(len(head).to_bytes(8, "big") + head + b"".join(a.tobytes() for a in arrays))


def recv(sock):
    def read(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(n - len(buf), 1 << 22))
            if not chunk:
                raise EOFError("connection closed")
            buf += chunk
        return buf

    head = json.loads(bytes(read(int.from_bytes(read(8), "big"))))

    def dec(x):
        if isinstance(x, list):
            return [dec(v) for v in x]
        if not isinstance(x, dict):
            return x
        if "__nd__" in x:
            dtype, shape = np.dtype(x["__nd__"][0]), x["__nd__"][1]
            return np.frombuffer(read(dtype.itemsize * int(np.prod(shape))), dtype).reshape(shape)  # writable
        if "__dc__" in x:
            from roboskills.eval import interface

            kind = getattr(interface, x.pop("__dc__"))
            return kind(**{k: dec(v) for k, v in x.items()})
        if "__fn__" in x:
            module, name = x["__fn__"].split(":")
            if not module.startswith("roboskills."):
                raise ValueError(f"refusing to import {module!r} from the wire")
            f = functools.reduce(getattr, name.split("."), importlib.import_module(module))
            return functools.partial(f, *dec(x["args"]), **dec(x["kw"])) if "args" in x else f
        return {k: dec(v) for k, v in x.items()}

    return dec(head)  # arrays are read in header order, which is the order `send` wrote them


class Sim:
    """The host's end of one simulator socket."""

    def __init__(self, port, timeout=1800):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.lost = None

    def request(self, **msg):
        if self.lost:
            raise RuntimeError(f"the simulator is gone ({self.lost})")
        try:
            send(self.sock, msg)
            rep = recv(self.sock)
        except (OSError, EOFError) as e:
            self.lost = repr(e)
            raise RuntimeError(f"the simulator is gone ({e!r})") from None
        if "error" in rep:
            raise RuntimeError(f"simulator: {rep['error']}")
        return rep

    def close(self):
        self.sock.close()
