import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the current runtime")
class ModelShapeTests(unittest.TestCase):
    def test_gnn_classifier_output_shape(self):
        import torch

        from src.models.gnn_classifier import GNNFraudClassifier

        features = torch.randn(8, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        model = GNNFraudClassifier(5, 16, layers=2, dropout=0.0)
        self.assertEqual(tuple(model(features, edge_index).shape), (8,))

    def test_hybrid_output_shapes(self):
        import torch

        from src.models.hybrid import HybridGNNAutoencoder

        features = torch.randn(8, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        model = HybridGNNAutoencoder(5, 16, 4, gnn_layers=2, dropout=0.0)
        outputs = model(features, edge_index)
        self.assertEqual(tuple(outputs["logits"].shape), (8,))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (8, 5))
        self.assertEqual(tuple(outputs["latent"].shape), (8, 4))
        self.assertEqual(tuple(outputs["graph_embedding"].shape), (8, 16))


if __name__ == "__main__":
    unittest.main()
