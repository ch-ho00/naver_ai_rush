from fire import Fire
import argparse

from spam.training.train import train

if __name__ == '__main__':
    Fire(train)
