import numpy as np
import pytest
from mlp import MLP

def test_forward_classification():
    in_channels = np.random.rand(3, 4)
    hidden_channels = np.random.rand(4, 2)
    X = np.random.rand(5, 3)
    mlp = MLP(in_channels, hidden_channels, 'ReLU', 'ReLU', task='classification')
    output = mlp.forward(X)
    assert output.shape == (5, 2), f"Expected output shape (5, 2), but got {output.shape}"

def test_forward_regression():
    in_channels = np.random.rand(3, 4)
    hidden_channels = np.random.rand(4, 1)
    X = np.random.rand(5, 3)
    mlp = MLP(in_channels, hidden_channels, 'ReLU', 'ReLU', task='regression')
    output = mlp.forward(X)
    assert output.shape == (5, 1), f"Expected output shape (5, 1), but got {output.shape}"

def test_backward():
    in_channels = np.random.rand(3, 4)
    hidden_channels = np.random.rand(4, 2)
    X = np.random.rand(5, 3)
    dL_dy = np.random.rand(5, 2)
    mlp = MLP(in_channels, hidden_channels, 'ReLU', 'ReLU', task='classification')
    mlp.forward(X)
    dL_din_channels, dL_dhidden_channels = mlp.backward(dL_dy)
    assert dL_din_channels.shape == in_channels.shape, f"Expected dL_din_channels shape {in_channels.shape}, but got {dL_din_channels.shape}"
    assert dL_dhidden_channels.shape == hidden_channels.shape, f"Expected dL_dhidden_channels shape {hidden_channels.shape}, but got {dL_dhidden_channels.shape}"

def test_predict_classification():
    in_channels = np.random.rand(3, 4)
    hidden_channels = np.random.rand(4, 2)
    X = np.random.rand(5, 3)
    mlp = MLP(in_channels, hidden_channels, 'ReLU', 'ReLU', task='classification')
    predictions = mlp.predict(X)
    assert predictions.shape == (5,), f"Expected predictions shape (5,), but got {predictions.shape}"

def test_predict_regression():
    in_channels = np.random.rand(3, 4)
    hidden_channels = np.random.rand(4, 1)
    X = np.random.rand(5, 3)
    mlp = MLP(in_channels, hidden_channels, 'ReLU', 'ReLU', task='regression')
    predictions = mlp.predict(X)
    assert predictions.shape == (5, 1), f"Expected predictions shape (5, 1), but got {predictions.shape}"

    class MLP():
        # ... (existing methods)

if __name__ == "__main__":
    pytest.main()