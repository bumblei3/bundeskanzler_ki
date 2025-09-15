"""
Global stubs for testing the Bundeskanzler-KI.
Provides minimal implementations of external libraries for testing.
"""

import math
import sys


# Minimal numpy stub for testing
class _NPStub:
    def exp(self, x):
        return math.exp(x)

    def isscalar(self, x):
        return isinstance(x, (int, float))

    def argmax(self, a, axis=None):
        # Simple argmax implementation for testing
        if isinstance(a, list):
            if axis is None:
                return a.index(max(a))
            else:
                # For simplicity, ignore axis for now
                return a.index(max(a))
        return 0

    # Add ndarray type alias and make it accessible
    ndarray = list

    # Add array function
    def array(self, data, dtype=None):
        # Return data as-is for testing, but ensure it's list-like
        if isinstance(data, list):
            return data
        return [data]

    # Add argsort function
    def argsort(self, a):
        # Simple argsort implementation for testing
        if isinstance(a, list):
            return sorted(range(len(a)), key=lambda i: a[i], reverse=True)
        return [0]


# Lightweight stub of tensorflow for testing
class _TFStub:
    # Add __spec__ to satisfy importlib checks
    __spec__ = type(
        "ModuleSpec", (), {"name": "tensorflow", "origin": None, "has_location": False}
    )()

    # minimal Tensor marker class for type annotations
    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

        def __getitem__(self, key):
            if isinstance(self.value, list) and isinstance(key, slice):
                return _TFStub.Tensor(self.value[key])
            elif isinstance(self.value, list):
                return _TFStub.Tensor(self.value[key])
            return _TFStub.Tensor(self.value)

        def __len__(self):
            if isinstance(self.value, list):
                return len(self.value)
            return 1

    # Add TensorShape stub
    class TensorShape:
        def __init__(self, dims):
            self.dims = dims

    def constant(self, x):
        # return a Tensor wrapper - can handle single values or lists
        return _TFStub.Tensor(x)

    def concat(self, arrays, axis=0):
        # naive concat: flatten list of lists along axis=0
        out = []
        for a in arrays:
            if hasattr(a, "value"):
                if isinstance(a.value, list):
                    out.extend(a.value)
                else:
                    out.append(a.value)
            else:
                if isinstance(a, list):
                    out.extend(a)
                else:
                    out.append(a)
        return _TFStub.Tensor(out)

    def timestamp(self):
        import time

        return time.time()

    # Add keras submodule stub
    class keras:
        class preprocessing:
            class text:
                class Tokenizer:
                    def __init__(self, **kwargs):
                        self.word_index = {}
                        self.index_word = {}

                    def fit_on_texts(self, texts):
                        pass

                    def texts_to_sequences(self, texts):
                        return [[1, 2, 3] for _ in texts]  # Mock sequences

            class sequence:
                @staticmethod
                def pad_sequences(
                    sequences,
                    maxlen=None,
                    dtype="int32",
                    padding="pre",
                    truncating="pre",
                    value=0.0,
                ):
                    # Return sequences as numpy arrays with padding
                    import numpy as np

                    if maxlen is None:
                        maxlen = max(len(seq) for seq in sequences) if sequences else 0
                    result = []
                    for seq in sequences:
                        if len(seq) < maxlen:
                            if padding == "post":
                                padded = seq + [value] * (maxlen - len(seq))
                            else:  # 'pre'
                                padded = [value] * (maxlen - len(seq)) + seq
                        else:
                            if truncating == "post":
                                padded = seq[:maxlen]
                            else:  # 'pre'
                                padded = seq[-maxlen:]
                        result.append(padded)
                    return np.array(result, dtype=dtype)

        class Model:
            pass

        class layers:
            class Dense:
                pass

            class Embedding:
                pass

        class callbacks:
            class EarlyStopping:
                def __init__(self, **kwargs):
                    pass

            class ReduceLROnPlateau:
                def __init__(self, **kwargs):
                    pass

            class TensorBoard:
                def __init__(self, **kwargs):
                    pass  # Apply stubs globally - COMMENTED OUT to avoid interfering with real implementations


# sys.modules.setdefault('tensorflow', _TFStub())
# sys.modules.setdefault('numpy', _NPStub())
# Make ndarray available as numpy.ndarray
# sys.modules['numpy'].ndarray = _NPStub.ndarray
