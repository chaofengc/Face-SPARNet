import torch
import torch.nn as nn
import torch.optim as optim

from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils
from models.sparnet import SPARNet

class SPARNetModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        parser.add_argument('--scale_factor', type=int, default=8, help='upscale factor for sparnet')
        parser.add_argument('--lambda_pix', type=float, default=1.0, help='weight for pixel loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = SPARNet(res_depth=opt.res_depth, norm_type=opt.Gnorm, att_name=opt.att_name, bottleneck_size=opt.bottleneck_size) 
        self.netG = networks.define_network(opt, self.netG)

        self.model_names = ['G']
        self.load_model_names = ['G']
        self.loss_names = ['Pix'] 
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G]

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

    def backward_G(self):
        # Pix loss
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        self.loss_Pix.backward()
    
    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
 
    def get_current_visuals(self, size=128):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]
        
        return visual_imgs





