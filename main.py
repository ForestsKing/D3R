import argparse

import torch

from exp.exp import Exp
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SMD', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of the data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path of the checkpoint')

    parser.add_argument('--itr', type=int, default=5, help='num of evaluation')
    parser.add_argument('--epochs', type=int, default=8, help='epoch of train')
    parser.add_argument('--patience', type=int, default=3, help='patience of early stopping')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of data')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

    parser.add_argument('--period', type=int, default=1440, help='approximate period of time series')
    parser.add_argument('--train_rate', type=float, default=0.8, help='rate of train set')
    parser.add_argument('--window_size', type=int, default=64, help='size of sliding window')

    parser.add_argument('--model_dim', type=int, default=512, help='dimension of hidden layer')
    parser.add_argument('--ff_dim', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--atten_dim', type=int, default=64, help='dimension of various attention')

    parser.add_argument('--block_num', type=int, default=2, help='num of various block')
    parser.add_argument('--head_num', type=int, default=8, help='num of attention head')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout')

    parser.add_argument('--time_steps', type=int, default=1000, help='time step of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='start of diffusion beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='end of diffusion beta')

    parser.add_argument('--t', type=int, default=500, help='time step of adding noise')
    parser.add_argument('--p', type=float, default=10.00, help='peak value of trend disturbance')
    parser.add_argument('--d', type=int, default=30, help='shift of period')

    parser.add_argument('--q', type=float, default=0.01, help='init anomaly probability of spot')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='device ids of gpus')

    config = vars(parser.parse_args())
    setSeed(config['random_seed'])
    torch.cuda.set_device(config['gpu_id'])

    for ii in range(config['itr']):
        exp = Exp(config)
        exp.train()
        exp.test()
