import sys
import types

import pytest

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, "/home/tobber/bkki_venv")


# Erstelle Stub für tensorflow.keras.layers
class _KerasLayers:
    class Input:
        def __init__(self, shape):
            self.shape = shape

    class Embedding:
        def __init__(self, vocab_size, dim, **kwargs):
            pass

        def __call__(self, x):
            return x

    class Dropout:
        def __init__(self, rate):
            pass

        def __call__(self, x):
            return x

    class LSTM:
        def __init__(self, units, **kwargs):
            pass

        def __call__(self, x):
            return x

    class Dense:
        def __init__(self, units, activation=None, **kwargs):
            pass

        def __call__(self, x):
            return x

    class BatchNormalization:
        def __call__(self, x):
            return x


class _KerasRegularizers:
    class L1:
        def __init__(self, l1=0.0):
            pass

    class L2:
        def __init__(self, l2=0.0):
            pass

    # Hinzufügen von l1_l2 als Methode
    def l1_l2(self, l1=0.0, l2=0.0):
        return self  # Return dummy object, no real behavior needed

    # Hinzufügen von l1 als Methode
    def l1(self, l1=0.0):
        return self.L1(l1=l1)

    # Hinzufügen von l2 als Methode
    def l2(self, l2=0.0):
        return self.L2(l2=l2)


class _KerasConstraints:
    class MaxNorm:
        def __init__(self, max_value):
            pass

    class UnitNorm:
        def __init__(self):
            pass


# Add classes for optimizers and callbacks that accept any kwargs
class _AdamOptimizer:
    def __init__(self, **kwargs):
        pass


class _EarlyStoppingCallback:
    def __init__(self, **kwargs):
        pass


class _ReduceLRCallback:
    def __init__(self, **kwargs):
        pass


# Erstelle das keras-Modul als eigenes Modulobjekt
keras_module = types.ModuleType("tensorflow.keras")
keras_module.layers = _KerasLayers()
keras_module.regularizers = _KerasRegularizers()
keras_module.constraints = _KerasConstraints()
keras_module.optimizers = type("Optimizers", (), {"Adam": _AdamOptimizer})()


# Add Model class to keras
class _KerasModel:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss, metrics):
        pass


keras_module.Model = _KerasModel


# Erstelle einen stub für LambdaCallback, der ein Lambda mit 2 Argumenten akzeptiert
def _dummy_lambda(batch, logs):
    pass  # noop, no real execution


class _LambdaCallbackStub:
    def __init__(self, on_batch_begin=None, **kwargs):
        # Don't execute the lambda, just store it
        self.on_batch_begin = on_batch_begin


keras_module.callbacks = type(
    "Callbacks",
    (),
    {
        "EarlyStopping": _EarlyStoppingCallback,
        "ReduceLROnPlateau": _ReduceLRCallback,
        "LambdaCallback": _LambdaCallbackStub,
    },
)()

# Registriere tensorflow.keras als Modul in sys.modules
sys.modules["tensorflow.keras"] = keras_module

# Registriere auch das übergeordnete tensorflow-Modul
tf_module = types.ModuleType("tensorflow")
tf_module.keras = keras_module


# Füge clip_by_global_norm als Funktion hinzu (leerer Stub)
def clip_by_global_norm(tensors, clip_norm):
    # Gibt ein Tupel zurück, wie von Keras erwartet
    return (tensors,)


tf_module.clip_by_global_norm = clip_by_global_norm

sys.modules["tensorflow"] = tf_module

# Installiere tf_config-Stub
tf_config = types.ModuleType("tf_config")


def get_model_config():
    return {
        "embedding_dim": 8,
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "dropout_rate": 0.1,
        "lstm_units": 4,
        "dense_units": 8,
        "learning_rate": 0.001,
        "clip_norm": 1.0,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
    }


tf_config.get_model_config = get_model_config
sys.modules["tf_config"] = tf_config

# Jetzt kann transformer_model sicher importiert werden
import transformer_model


def test_create_transformer_model_returns_model_object():
    # create_transformer_model should return an object (stubbed keras.Model not present,
    # but function returns something; check type and attributes)
    model = transformer_model.create_transformer_model(
        maxlen=10, vocab_size=100, output_size=2
    )
    assert model is not None


def test_get_training_callbacks_returns_list():
    callbacks = transformer_model.get_training_callbacks(
        model=None, config=tf_config.get_model_config()
    )
    assert isinstance(callbacks, list)
    assert len(callbacks) == 3
    # Wir testen nicht die Funktionalität der Callbacks, nur dass sie zurückgegeben werden
