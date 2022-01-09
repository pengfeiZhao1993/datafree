from __future__ import absolute_import, division, print_function
import sys
sys.path.append('..')

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.super_generator import SuperGenerator, init_weights
from utils.space_util import sample_subnet_config
from utils.measure_util import get_dummy_input
from configs.config import args, logger


init_set = ['linear1', 'bn1', 'bn2', 'bn3', 'conv1', 'conv2', 'conv3']


def main():
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    device_using = device

    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                args.local_rank, device_using, args.n_gpu, bool(args.local_rank != -1))

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model = SuperGenerator(256, 64, 32, 3)
    print(model)
    model.apply(init_weights)
    model.to(device_using)

    signal = get_dummy_input(device_using)

    for _ in range(100):
        sample_config = sample_subnet_config()
        model.set_sample_config(sample_config)
        img = model(signal)
        ret = img.permute(2, 3, 1, 0).squeeze()
        plt.imshow(ret.cpu().detach().numpy())
        plt.show()


if __name__ == "__main__":
    main()






