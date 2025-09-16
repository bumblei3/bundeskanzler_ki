import sys
import types

import pytest

# Ensure local module is importable
sys.path.insert(0, "/home/tobber/bkki_venv")


# Stub numpy and TF for transfer_learning
class _NPStub:
    def array(self, x):
        return x


class _TFLayers:
    class Dense:
        def __init__(self, units, activation=None):
            self.units = units
            self.activation = activation

        def __call__(self, x):
            return x

    class Dropout:
        def __init__(self, rate):
            self.rate = rate

        def __call__(self, x):
            return x

    class LayerNormalization:
        def __call__(self, x):
            return x


class _TFKeras:
    layers = _TFLayers()

    class Sequential:
        def __init__(self, layers):
            self.layers = layers
            self.trainable_variables = []

        def __call__(self, x):
            return x

    class optimizers:
        class Adam:
            def __init__(self, learning_rate=0.001):
                self.learning_rate = learning_rate

            def apply_gradients(self, grads_and_vars):
                pass

        class Optimizer:
            def apply_gradients(self, grads_and_vars):
                pass

    class losses:
        @staticmethod
        def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False):
            return _MockTensor([0.5])  # dummy loss as tensor


class _MockTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value

    def __add__(self, other):
        if isinstance(other, _MockTensor):
            if isinstance(self.value, list) and isinstance(other.value, list):
                return _MockTensor([self.value[0] + other.value[0]])
            return _MockTensor(self.value + other.value)
        # Handle adding float to list tensor
        if isinstance(self.value, list):
            return _MockTensor([self.value[0] + other])
        return _MockTensor(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __float__(self):
        if isinstance(self.value, list):
            return float(self.value[0])
        return float(self.value)


class _TFStub:
    # Type annotation placeholder
    class Tensor:
        pass

    keras = _TFKeras()

    @staticmethod
    def reduce_mean(tensor, axis=None):
        return _MockTensor([0.5, 0.3])  # dummy pooled embeddings

    @staticmethod
    def reduce_max(tensor, axis=None):
        return _MockTensor([0.8])

    @staticmethod
    def argmax(tensor, axis=None):
        return _MockTensor([0])

    class nn:
        @staticmethod
        def softmax(logits, axis=-1):
            return [[0.8, 0.2]]

        @staticmethod
        def l2_loss(tensor):
            return 0.01

    @staticmethod
    def add_n(losses):
        return 0.05

    class GradientTape:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def gradient(self, loss, variables):
            return [0.1] * len(variables) if variables else []


sys.modules["tensorflow"] = _TFStub()
sys.modules["numpy"] = _NPStub()

# Mock logging
import types

logging_module = types.ModuleType("logging")
logging_module.basicConfig = lambda **kwargs: None
logging_module.INFO = 20
logging_module.getLogger = lambda name: types.SimpleNamespace(info=lambda msg: None)
logging_module.info = lambda msg: None  # Add missing info function
logging_module.warning = lambda msg: None  # Add missing warning function
logging_module.error = lambda msg: None  # Add missing error function
sys.modules["logging"] = logging_module

# Now import the module
import transfer_learning


def test_transfer_learner_init():
    """Test TransferLearner initialization."""
    learner = transfer_learning.TransferLearner(num_classes=3, embedding_dim=128, hidden_dim=64)
    assert learner.num_classes == 3
    assert learner.embedding_dim == 128
    assert learner.classifier is not None
    assert learner.adapter is not None


def test_adapt_method():
    """Test the adapt method returns adapted embeddings."""
    learner = transfer_learning.TransferLearner(num_classes=2, embedding_dim=64)
    # Mock embeddings as a simple list
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    labels = [0, 1]  # Required labels parameter

    # adapt should return something (our stub returns input)
    result = learner.adapt(embeddings, labels)
    assert result is not None


def test_classify_method():
    """Test the classify method returns category and confidence."""
    learner = transfer_learning.TransferLearner(num_classes=2, embedding_dim=64)

    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    patterns = {
        "politics": {"keywords": ["policy", "government"]},
        "economy": {"keywords": ["market", "finance"]},
    }

    category, confidence = learner.classify(embeddings, patterns)

    assert isinstance(category, str)
    assert category in patterns.keys()
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_train_step():
    """Test the train_step method returns a loss value."""
    learner = transfer_learning.TransferLearner(num_classes=2, embedding_dim=64)

    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    labels = [0, 1]

    # Mock optimizer
    optimizer = _TFKeras.optimizers.Optimizer()

    loss = learner.train_step(embeddings, labels, optimizer)

    assert isinstance(loss, float)
    assert loss >= 0.0
