import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.models.autoencoder import Autoencoder
from src.models.gnn_classifier import GNNFraudClassifier
from src.models.hybrid import HybridGNNAutoencoder
from src.training.torch_trainer import fit_autoencoder, fit_gnn, fit_hybrid


def test_autoencoder_restores_minimum_validation_loss():
    torch.manual_seed(42)
    x = torch.randn(12, 3)
    y = torch.tensor([0, 0, 1]*4)
    val = torch.randn(9, 3)
    labels = torch.tensor([0, 0, 1]*3)
    model = Autoencoder(3, 4, 2)
    result = fit_autoencoder(model, x, y, 3, .01, val, labels, batch_size=4)
    assert result.best_epoch == 1 + np.argmin([r['validation_loss'] for r in result.history])
    with torch.no_grad():
        normal = val[labels == 0]
        loss = torch.mean((model(normal)[0]-normal)**2).item()
    assert abs(loss-result.best_validation_loss) < 1e-6


def test_supervised_models_restore_maximum_validation_ap():
    torch.manual_seed(42)
    x = torch.randn(12, 3)
    y = torch.tensor([0, 0, 1]*4)
    context = torch.cat([x, torch.randn(6, 3)])
    val_y = torch.tensor([0, 1, 0, 1, 0, 1])
    edges = torch.empty((2, 0), dtype=torch.long)
    for hybrid in [False, True]:
        model = HybridGNNAutoencoder(3, 4, 2, dropout=0.) if hybrid else GNNFraudClassifier(3, 4, dropout=0.)
        args = (model, x, edges, y, context, edges, val_y, 12, 3, .01, .0001)
        result = fit_hybrid(*args, .01, 5) if hybrid else fit_gnn(*args, 5)
        assert result.best_epoch == 1 + np.argmax([r['validation_ap'] for r in result.history])
        with torch.no_grad():
            output = model(context, edges)
            scores = (output['logits'] if hybrid else output)[12:].numpy()
        assert average_precision_score(val_y.numpy(), scores) == max(r['validation_ap'] for r in result.history)
