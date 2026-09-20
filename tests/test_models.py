import torch

from cognition_ake.models import METHODS, CognitionAKETagger


def test_all_controlled_models_forward_and_decode():
    config = {"embedding_dim": 16, "hidden_size": 8, "alignment_hidden": 4, "lstm_layers": 1, "dropout": 0.1, "cib_hidden": 6, "cib_latent_dim": 3}
    batch, length = 2, 5
    embedding = torch.randn(batch, length, 16)
    cognitive = torch.randn(batch, length, 5)
    feature_mask = torch.ones(batch, length, 5)
    valid = torch.ones(batch, length)
    lengths = torch.tensor([5, 3])
    token_mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
    for method in METHODS:
        model = CognitionAKETagger(method, config)
        output = model.forward_all(embedding, cognitive, feature_mask, valid, lengths, token_mask)
        assert output["emissions"].shape == (batch, length, 3)
        assert [len(tags) for tags in model.decode(output["emissions"], token_mask)] == [5, 3]

