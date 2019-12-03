import torch
import torch.nn.functional as F 
import os, sys
import yaml, time
from model import Net
from utils import get_loader

with open(sys.argv[1]) as f:
    config = yaml.load(f, Loader=yaml.Loader)

print('Preparing datasets...')
csv_train = os.path.join(config['csv_folder'], 'train.csv')
csv_valid = os.path.join(config['csv_folder'], 'valid.csv')
if_label = True
loader_train, num_imgs_train = get_loader(if_label, config['imgs_folder'], csv_train, config['resize_size'], 
                                            config['crop_size'], True, config['batch_size'], config['num_workers'])
loader_valid, num_imgs_valid = get_loader(if_label, config['imgs_folder'], csv_valid, config['resize_size'], 
                                            config['crop_size'], False, config['batch_size']//5, config['num_workers'])
print('{} training images loaded.\n{} validation images loaded.'.format(num_imgs_train, num_imgs_valid))

print('Constructing network...')
net = Net(config)
device = 'cuda:%d' % config['device'] if torch.cuda.is_available() else 'cpu' 
net.to(device)
net.ready()

print('Train start!')
tot_epoch = config['epoch']
for step in range(tot_epoch):
    print(time.asctime(time.localtime(time.time())))
    print('Epoch: %d' % (step+1))
    print('  LR:', net.scheduler.get_lr()[0])
    
    tot_loss = 0.
    tot_acc = 0.
    for i, (x, label) in enumerate(loader_train):
        x, label = x.to(device), label.to(device)
        net.optimizer.zero_grad()
        loss, acc = net(x, label)
        loss, acc = torch.mean(loss), torch.mean(acc)
        loss.backward()
        net.optimizer.step()
        print('  Batch: %d, loss: %f, acc: %f' % (i, float(loss), float(acc)))
        with torch.no_grad():
            tot_loss = (i * tot_loss + float(loss)) / (i + 1)
            tot_acc = (i * tot_acc + float(acc)) / (i + 1)
    
    print('  Average loss: %f, average acc: %f' % (tot_loss, tot_acc))

    tot_loss = 0.
    tot_acc = 0.
    for i, (x, label) in enumerate(loader_valid):
        x, label = x.to(device), label.to(device)  # x is 5D tensor
        with torch.no_grad():
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)  # x is 4D now
            x = net(x)  # x is 2D now
            x = x.view(b, ncrops, -1)  # x is 3D now
            x = torch.mean(x, dim=1, keepdim=False)  # x is 2D again
            pred = torch.argmax(x, dim=1)
            loss = F.nll_loss(x, label, reduction='mean')
            acc = (pred == label).type(torch.float32).mean()
            tot_loss = (i * tot_loss + float(loss)) / (i + 1)
            tot_acc = (i * tot_acc + float(acc)) / (i + 1)
    print('  validation: loss: %f, acc: %f' % (float(tot_loss), float(tot_acc)))
    
    net.scheduler.step()
    net.save(step+1)

print('Train finished!')
