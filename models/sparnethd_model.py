import torch
import torch.nn as nn
import torch.optim as optim
import copy

from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils
from models.sparnet import SPARNet

class SPARNetHDModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--lambda_pcp', type=float, default=1.0, help='weight for vgg perceptual loss')
            parser.add_argument('--lambda_pix', type=float, default=100.0, help='weight for pixel loss')
            parser.add_argument('--lambda_fm', type=float, default=10.0, help='weight for sr')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for sr')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = SPARNet(min_ch=32, max_ch=512, in_size=512, out_size=512, min_feat_size=32, 
                            res_depth=opt.res_depth, norm_type=opt.Gnorm, att_name=opt.att_name, bottleneck_size=opt.bottleneck_size) 
        self.netG = networks.define_network(opt, self.netG, use_norm='spectral_norm')

        self.netH = copy.deepcopy(self.netG)

        self.model_names = ['G', 'D', 'H']
        self.load_model_names = ['G', 'H']
        self.loss_names = ['Pix', 'PCP', 'G', 'FM', 'D'] # Generator loss, fm loss, parsing loss, discriminator loss
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']

        if self.isTrain:
            self.load_model_names = ['G', 'D', 'H']

            self.netD = networks.MultiScaleDiscriminator(3, n_layers=opt.n_layers_D, norm_type=opt.Dnorm, num_D=opt.num_D)
            self.netD = networks.define_network(opt, self.netD, use_norm='spectral_norm') 
            self.vgg19 = loss.PCPFeat('./pretrain_models/vgg19-dcbb9e9d.pth', 'vgg')
            self.vgg19 = networks.define_network(opt, self.vgg19, isTrain=False, init_network=False)

            self.criterionFM = loss.FMLoss().to(opt.data_device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.99))
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def load_pretrain_model(self,):
        print('Loading pretrained model', self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.module.load_state_dict(weight)
    
    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(self.opt.data_device)
        self.img_HR = input['HR'].to(self.opt.data_device)

    def forward(self):
        self.img_SR = self.netG(self.img_LR) 

        self.real_D_results = self.netD(self.img_HR, return_feat=True)
        self.fake_D_results = self.netD(self.img_SR.detach(), return_feat=False)
        self.fake_G_results = self.netD(self.img_SR, return_feat=True)

        self.fake_vgg_feat = self.vgg19(self.img_SR)
        self.real_vgg_feat = self.vgg19(self.img_HR)

        with torch.no_grad():
            self.accumulate(self.netH, self.netG)
            self.ema_img_SR = self.netH(self.img_LR)

    def backward_G(self):
        # Pix loss
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        # perceptual loss
        self.loss_PCP = self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat) * self.opt.lambda_pcp 

        # Feature matching loss
        tmp_loss =  0
        for i in range(self.opt.num_D):
            tmp_loss = tmp_loss + self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1]) 
        self.loss_FM = tmp_loss * self.opt.lambda_fm / self.opt.num_D

        # Generator loss
        tmp_loss = 0
        for i in range(self.opt.num_D):
            tmp_loss = tmp_loss + self.criterionGAN(self.fake_G_results[i][0], True, for_discriminator=False)
        self.loss_G = tmp_loss * self.opt.lambda_g / self.opt.num_D

        loss = self.loss_Pix + self.loss_PCP + self.loss_FM + self.loss_G
        loss.backward()

    def backward_D(self, ):

        loss = 0
        for i in range(self.opt.num_D):
            loss += 0.5 * (self.criterionGAN(self.fake_D_results[i], False) + self.criterionGAN(self.real_D_results[i][0], True))
        self.loss_D = loss / self.opt.num_D 
        self.loss_D.backward()
    
    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D ------------
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_visuals(self, size=512):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.ema_img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]
        
        return visual_imgs

