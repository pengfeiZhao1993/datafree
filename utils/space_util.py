import random


def get_subnet_search_space():
    config = {
        'conv1_sample_size': [16, 32, 48, 64],
        'conv2_sample_size': [16, 32, 48, 64],
        'conv3_sample_size': [16, 32, 48, 64],
        'conv1_kernel_size': [1, 3, 5, 7],
        'conv2_kernel_size': [1, 3, 5, 7],
        'conv3_kernel_size': [1, 3, 5, 7],
    }
    return config


def sample_subnet_config(reset_rand_seed=False):
    if reset_rand_seed:
        random.seed(0)

    search_space = get_subnet_search_space()
    config = {}
    config['conv1_sample_size'] = random.choice(search_space['conv1_sample_size'])
    config['conv2_sample_size'] = random.choice(search_space['conv2_sample_size'])
    config['conv3_sample_size'] = random.choice(search_space['conv3_sample_size'])
    config['conv1_kernel_size'] = random.choice(search_space['conv1_kernel_size'])
    config['conv2_kernel_size'] = random.choice(search_space['conv2_kernel_size'])
    config['conv3_kernel_size'] = random.choice(search_space['conv3_kernel_size'])
    return config


def get_default_config(config):
    sample_config = {}
    sample_config['conv1_sample_size'] = config.ngf
    sample_config['conv2_sample_size'] = config.ngf
    sample_config['conv3_sample_size'] = config.ngf
    sample_config['conv1_kernel_size'] = 3
    sample_config['conv2_kernel_size'] = 3
    sample_config['conv3_kernel_size'] = 3
    return sample_config







