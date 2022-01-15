import torch
import torch.nn as nn

from modules.superconv2d import SuperConv2d
from modules.superlinear import SuperLinear
from modules.superbatchnorm import SuperBatchNorm2d


class SuperGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(SuperGenerator, self).__init__()

        self.super_conv1_ngf = ngf
        self.super_conv2_ngf = ngf
        self.super_conv3_ngf = ngf
        self.super_conv1_ks = 7
        self.super_conv2_ks = 7
        self.super_conv3_ks = 7

        self.sample_conv1_ngf = None
        self.sample_conv2_ngf = None
        self.sample_conv3_ngf = None
        self.sample_conv1_ks = None
        self.sample_conv2_ks = None
        self.sample_conv3_ks = None

        self.sample_config = None

        self.nz = nz
        self.nc = nc
        self.init_size = img_size // 4
        self.resolution = self.init_size ** 2

        self.linear1 = SuperLinear(nz, self.super_conv1_ngf * 2 * self.resolution)
        self.l1 = nn.Sequential(self.linear1)

        self.conv1 = SuperConv2d(self.super_conv1_ngf * 2, self.super_conv2_ngf * 2, self.super_conv1_ks, stride=1,
                                 padding=self.super_conv1_ks//2, bias=False)
        self.conv2 = SuperConv2d(self.super_conv2_ngf * 2, self.super_conv3_ngf * 2, self.super_conv2_ks, stride=1,
                                 padding=self.super_conv2_ks//2, bias=False)
        self.conv3 = SuperConv2d(self.super_conv3_ngf * 2, nc, self.super_conv3_ks, stride=1,
                                 padding=self.super_conv3_ks//2, bias=False)

        self.bn1 = SuperBatchNorm2d(self.super_conv1_ngf * 2)
        self.bn2 = SuperBatchNorm2d(self.super_conv2_ngf * 2)
        self.bn3 = SuperBatchNorm2d(self.super_conv3_ngf * 2)

        self.conv_blocks = nn.Sequential(
            self.bn1,
            nn.Upsample(scale_factor=2),
            self.conv1,
            self.bn2,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            self.conv2,
            self.bn3,
            nn.LeakyReLU(0.2, inplace=True),
            self.conv3,
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def set_sample_config(self, sample_config):

        self.sample_config = sample_config
        self.sample_conv1_ngf = sample_config['conv1_sample_size']
        self.sample_conv2_ngf = sample_config['conv2_sample_size']
        self.sample_conv3_ngf = sample_config['conv3_sample_size']
        self.sample_conv1_ks = sample_config['conv1_kernel_size']
        self.sample_conv2_ks = sample_config['conv2_kernel_size']
        self.sample_conv3_ks = sample_config['conv3_kernel_size']

        self.linear1.set_sample_config(self.nz, self.sample_conv1_ngf * 2 * self.resolution)

        self.conv1.set_sample_config(self.sample_conv1_ngf * 2, self.sample_conv2_ngf * 2, self.sample_conv1_ks)
        self.conv2.set_sample_config(self.sample_conv2_ngf * 2, self.sample_conv3_ngf * 2, self.sample_conv1_ks)
        self.conv3.set_sample_config(self.sample_conv3_ngf * 2, self.nc, self.sample_conv1_ks)

        self.bn1.set_sample_config(self.sample_conv1_ngf * 2)
        self.bn2.set_sample_config(self.sample_conv2_ngf * 2)
        self.bn3.set_sample_config(self.sample_conv3_ngf * 2)

    # wait to modify the name.split()
    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):

                if name == 'classifier':
                    continue

                if name.split('.')[1] == 'encoder' and eval(name.split('.')[3]) + 1 > config['common'][
                    'bert_layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())
        return sum(numels)

    def profile(self, mode=True):
        for module in self.modules():
            if hasattr(module, 'profile') and self != module:
                module.profile(mode)


def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (SuperConv2d, SuperLinear)):

        module.weight.data.normal_(mean=0.0, std=1.0)
    elif isinstance(module, SuperBatchNorm2d):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, SuperLinear) and module.bias is not None:
        module.bias.data.zero_()