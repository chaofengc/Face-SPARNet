from models.blocks import *
import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler

import torch.nn.utils as tutils

def apply_norm(net, weight_norm_type):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if weight_norm_type.lower() == 'spectral_norm':
                tutils.spectral_norm(m)
            elif weight_norm_type.lower() == 'weight_norm':
                tutils.weight_norm(m)
            else:
                pass


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if isinstance(net, nn.DataParallel):
        network_name = net.module.__class__.__name__
    else:
        network_name = net.__class__.__name__

    print('initialize network %s with %s' % (network_name, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net 


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(opt, net, isTrain=True, use_norm='none', init_network=True):
    apply_norm(net, use_norm)
    if not isTrain:
        net.eval()  
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(opt.device)
        net = torch.nn.DataParallel(net, opt.gpu_ids, output_device=opt.data_device)
    if init_network:
        init_weights(net, init_type='normal', init_gain=0.02)
    return net


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_ch, base_ch=64, n_layers=4, norm_type='none', relu_type='LeakyReLU', num_D=4):
        super().__init__()

        self.D_pool = nn.ModuleList()
        for i in range(num_D):
            netD = NLayerDiscriminator(input_ch, base_ch, depth=n_layers, norm_type=norm_type, relu_type=relu_type)
            self.D_pool.append(netD)

        def downsample(x):
            return nn.functional.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=False)
        self.downsample = downsample

    def forward(self, input, return_feat=False):
        results = []
        for netd in self.D_pool:
            output = netd(input, return_feat) 
            results.append(output)
            input = self.downsample(input)
        return results


class NLayerDiscriminator(nn.Module):
    def __init__(self,
            input_ch = 3,
            base_ch = 64,
            max_ch = 512,
            depth = 4,
            norm_type = 'none',
            relu_type = 'LeakyReLU',
            ):
        super().__init__()

        nargs = {'norm_type': norm_type, 'relu_type': relu_type}

        self.model = []
        self.model.append(ConvLayer(input_ch, base_ch, norm_type='none', relu_type=relu_type))
        for i in range(depth):
            cin  = min(base_ch * 2**(i), max_ch)
            cout = min(base_ch * 2**(i+1), max_ch)
            self.model.append(ConvLayer(cin, cout, scale='down', **nargs))
        self.model = nn.Sequential(*self.model)
        self.score_out = ConvLayer(cout, 1, use_pad=False)

    def forward(self, x, return_feat=False):
        ret_feats = []
        for idx, m in enumerate(self.model):
            x = m(x)
            ret_feats.append(x)
        x = self.score_out(x)
        if return_feat:
            return x, ret_feats
        else:
            return x

