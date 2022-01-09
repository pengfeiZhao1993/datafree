import torch
from torch import nn
from torch.nn import functional as F


class SuperConv2d(nn.Conv2d):

    def __init__(self, super_in_dim, super_out_dim, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SuperConv2d, self).__init__(super_in_dim, super_out_dim, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)

        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        super().reset_parameters()
        self.profiling = False

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.conv2d(x, self.samples['weight'], self.samples['bias'], self.stride, self.padding, self.dilation, self.groups)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def profile(self, mode=True):
        self.profiling = mode


def sample_weight(weight, sample_in_dim, sample_out_dim):

    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]
    return sample_weight


def sample_bias(bias, sample_out_dim):

    sample_bias = bias[:sample_out_dim]
    return sample_bias
