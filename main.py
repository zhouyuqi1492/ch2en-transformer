import torch
import math
import argparse

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--seed', type=int, default=2025)

# model parameters
parser.add_argument('--num_steps', type=int, default=128)
parser.add_argument('--hidden_num', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer',
                    choices=['sgd', 'adam', 'adamax'],
                    default='adam')

# Load dataset

# Trainer

# Evaluator

if __name__ == '__main__':
    pass
    # params init
    # input
    # output
