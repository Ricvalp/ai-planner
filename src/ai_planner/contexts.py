"""Stable hashes include values, shapes and versions, rather than mutable identities."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import numpy as np


def canonical(value):
    if is_dataclass(value):
        return canonical(asdict(value))
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if hasattr(value, "shape"):
        a = np.asarray(value)
        return {"shape": list(a.shape), "dtype": str(a.dtype), "values": a.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def frozen_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()[:20]
