"""Shared offline test helpers; production code must not import tests."""
import torch

class FixtureTokenizer:
    name = 'offline'
    eot_token = 50256

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]



def state(path):
    return torch.load(path / 'state.pt', weights_only=False, map_location='cpu')


# Smallest model shape used across math/optim tests (vocab 97 padded to 128).
TINY_MODEL = dict(layers=2, d_model=32, heads=2, d_key=8, d_value=16, d_mlp=48, vocab_size=97, padded_vocab=128)
