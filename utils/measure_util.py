# import torchprofile
import torch
import random
import numpy as np


def measure_flops(model, config, dummy_input):

    model.set_sample_config(config)
    model.profile(mode=True)
    macs = torchprofile.profile_macs(model, dummy_input)
    model.profile(mode=False)
    return macs*2


def get_dummy_input(device_using):

    dummy_input = torch.randn(1, 256).to(device_using)  # input_id

    return dummy_input