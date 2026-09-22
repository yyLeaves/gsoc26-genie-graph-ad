"""Train a graph autoencoder: python -m scripts.train_graph_ae --help."""

from src.training.options import parse_args
from src.training.trainer import train


if __name__ == "__main__":
    train(parse_args())
