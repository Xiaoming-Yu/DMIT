import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn  import functional as F
from torch.nn.utils.spectral_norm  import spectral_norm

class _AdaINorm(_BatchNorm):
    def __init__(self, num_features, num_con=8, eps=1e-5, momentum=0.1, track_running_stats=False):
        super(_AdaINorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_con = num_con
        if num_con >0:
            self.ConAlpha = spectral_norm(nn.Linear(num_con, num_features))
            self.ConBeta = spectral_norm(nn.Linear(num_con, num_features))
        
    def _check_input_dim(self, input):
        raise NotImplementedError
        
    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_AdaINorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
            
    def forward(self, input, ConInfor):
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)
        out = F.instance_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats, self.momentum, self.eps)
        
        if self.num_con >0:
            weight = self.ConAlpha(ConInfor).view(b,c,1,1)
            bias = self.ConBeta(ConInfor).view(b,c,1,1)
        else:
            weight = 1
            bias = 0
        return out.view(b, c, *input.size()[2:])*weight + bias


class AdaINorm2d(_AdaINorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))