import torch
import torch.nn.functional as F


class SuperBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, super_embed_dim, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(super_embed_dim, eps, momentum, affine)

        self.super_embed_dim = super_embed_dim
        self.sample_embed_dim = None
        self.samples = {}
        self.profiling = False

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        self.samples['running_mean'] = self.running_mean[:self.sample_embed_dim]
        self.samples['running_var'] = self.running_var[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        input = x
        if x.shape[1] != self.num_features:
            padding = torch.zeros([x.shape[0], self.num_features - x.shape[1], x.shape[2], x.shape[3]], device=x.device)
            input = torch.cat([input, padding], dim=1)


        ret = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)
        return ret[:, :x.shape[1]]
        # self.sample_parameters()
        # return F.batch_norm(x, running_mean=self.samples['running_mean'], running_var=self.samples['running_var'],
        #                     weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def profile(self, mode=True):
        self.profiling = mode