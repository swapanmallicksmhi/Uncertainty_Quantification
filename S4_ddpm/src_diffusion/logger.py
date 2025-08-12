"""
Logging utilities for multiple output formats.
"""

import os
import sys
import json
import time
import datetime
import tempfile
import warnings
import os.path as osp
from collections import defaultdict
from contextlib import contextmanager

# Logging levels
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

# ---------------------------
# Output Writers
# ---------------------------

class KVWriter:
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter:
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read")
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        key2str = {}
        for key, val in sorted(kvs.items()):
            valstr = f"{val:.3g}" if hasattr(val, "__float__") else str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if not key2str:
            print("WARNING: tried to write empty key-value dict")
            return

        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))
        dashes = "-" * (keywidth + valwidth + 7)

        lines = [dashes]
        for key, val in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(f"| {key:<{keywidth}} | {val:<{valwidth}} |")
        lines.append(dashes)

        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def writeseq(self, seq):
        self.file.write(" ".join(map(str, seq)) + "\n")
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        extra_keys = list(set(kvs.keys()) - set(self.keys))
        if extra_keys:
            self.keys.extend(sorted(extra_keys))
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            self.file.write(self.sep.join(self.keys) + "\n")
            for line in lines[1:]:
                self.file.write(line.strip() + self.sep * len(extra_keys) + "\n")

        row = [str(kvs.get(k, "")) for k in self.keys]
        self.file.write(self.sep.join(row) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        from tensorflow.python import pywrap_tensorflow
        import tensorflow as tf

        self.tf = tf
        self.event_pb2 = event_pb2
        self.writer = pywrap_tensorflow.EventsWriter(
            compat.as_bytes(osp.join(osp.abspath(dir), "events"))
        )
        self.step = 1

    def writekvs(self, kvs):
        def summary_val(k, v):
            return self.tf.Summary.Value(tag=k, simple_value=float(v))

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        self.writer.Close()
        self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, f"log{log_suffix}.txt"))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, f"progress{log_suffix}.json"))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, f"progress{log_suffix}.csv"))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, f"tb{log_suffix}"))
    else:
        raise ValueError(f"Unknown format: {format}")


# ---------------------------
# Logging API
# ---------------------------

def logkv(key, val): get_current().logkv(key, val)
def logkv_mean(key, val): get_current().logkv_mean(key, val)
def logkvs(d): [logkv(k, v) for k, v in d.items()]
def dumpkvs(): return get_current().dumpkvs()
def getkvs(): return get_current().name2val
def log(*args, level=INFO): get_current().log(*args, level=level)
def debug(*args): log(*args, level=DEBUG)
def info(*args): log(*args, level=INFO)
def warn(*args): log(*args, level=WARN)
def error(*args): log(*args, level=ERROR)
def set_level(level): get_current().set_level(level)
def set_comm(comm): get_current().set_comm(comm)
def get_dir(): return get_current().get_dir()
record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    t0 = time.time()
    key = f"wait_{scopename}"
    try:
        yield
    finally:
        get_current().name2val[key] += time.time() - t0


def profile(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with profile_kv(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------
# Logger Backend
# ---------------------------

class Logger:
    CURRENT = None
    DEFAULT = None

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        count = self.name2cnt[key]
        self.name2val[key] = (self.name2val[key] * count + val) / (count + 1)
        self.name2cnt[key] += 1

    def dumpkvs(self):
        d = self.name2val.copy()
        if self.comm:
            d = mpi_weighted_mean(
                self.comm,
                {k: (v, self.name2cnt.get(k, 1)) for k, v in self.name2val.items()}
            )
            if self.comm.rank != 0:
                d["dummy"] = 1
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return d

    def log(self, *args, level=INFO):
        if self.level <= level:
            for fmt in self.output_formats:
                if isinstance(fmt, SeqWriter):
                    fmt.writeseq(map(str, args))

    def set_level(self, level): self.level = level
    def set_comm(self, comm): self.comm = comm
    def get_dir(self): return self.dir
    def close(self): [fmt.close() for fmt in self.output_formats]


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()
    return Logger.CURRENT


def get_rank_without_mpi_import():
    for var in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if var in os.environ:
            return int(os.environ[var])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    all_data = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum, name2count = defaultdict(float), defaultdict(float)
        for data in all_data:
            for name, (val, count) in data.items():
                try:
                    name2sum[name] += float(val) * count
                    name2count[name] += count
                except ValueError:
                    warnings.warn(f"Cannot average non-numeric {name}={val}")
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    return {}


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR") or osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
        )

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix += f"-rank{rank:03d}"

    format_strs = format_strs or (
        os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        if rank == 0 else
        os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    )
    format_strs = list(filter(None, format_strs))
    output_formats = [make_output_format(fmt, dir, log_suffix) for fmt in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log(f"Logging to {dir}")


def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Logger reset to default")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prev_logger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prev_logger
