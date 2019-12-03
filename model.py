import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F 
import os
from utils import RAdam

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.backbone = models.resnet50(pretrained=False, num_classes=29)
        if config['is_train']:
            self._init_weights()
            self._get_optim()
        else:
            self.resume()

    def forward(self, x, label=None):
        x = self.backbone(x) # N*D        
        x = F.log_softmax(x, dim=1)

        if label is not None:
            loss = F.nll_loss(x, label, reduction='none') # N
            with torch.no_grad():
                pred = torch.argmax(x, dim=1)
                acc = (pred == label).type(torch.float32)
            return loss, acc        
        else:
            return x

    def ready(self):
        if self.config['is_train']:
            self.train()
        else:
            self.eval()

    def resume(self):
        folder = self.config['ckpt_dir']
        step = self.config['ckpt_step']
        self.load_state_dict(torch.load(os.path.join(folder, 'model_%d.ckpt'%step)))

    def save(self, step):
        folder = self.config['ckpt_dir']
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), os.path.join(folder, 'model_%d.ckpt'%step))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.)
                nn.init.constant_(m.bias.data, 0.)

    def _get_optim(self):
        #self.optimizer = optim.SGD(self.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], 
        #                            weight_decay=self.config['weight_decay'])
        self.optimizer = RAdam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        s = self.config['lr_steps'].split(',')
        milestones = []
        for i in s:
            milestones.append(int(i))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, 
                                                        gamma=self.config['lr_gamma'])
