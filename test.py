import torch
import torch.nn.functional as F
import os, sys, csv
import yaml, time
from model import Net
from utils import get_loader

with open(sys.argv[1]) as f:
    config = yaml.load(f, Loader=yaml.Loader)

print('Preparing datasets...')
loader, names = get_loader(if_label=False, imgs_folder=config['imgs_folder'], 
                            csv_file=config['csv_file'], resize_size=config['resize_size'], 
                            crop_size=config['crop_size'], is_train=config['is_train'], 
                            batch_size=config['batch_size'], num_workers=config['num_workers'])

print('Constructing network...')
net = Net(config)
device = 'cuda:%d' % config['device'] if torch.cuda.is_available() else 'cpu' 
net.to(device)
net.ready()

print('Test start!')
with torch.no_grad():
    tot_pred = None
    for i, x in enumerate(loader):
        x = x.to(device)  # x is 5D tensor
        b, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)  # x is 4D now
        x = net(x)  # x is 2D now
        x = x.view(b, ncrops, -1)  # x is 3D now
        x = torch.mean(x, dim=1, keepdim=False)  # x is 2D again
        pred = torch.argmax(x, dim=1)
        if tot_pred is None:
            tot_pred = pred.view(-1)
        else:
            tot_pred = torch.cat([tot_pred, pred.view(-1)])

print('Test finished!\nWriting to csv file...')

with open(os.path.join(config['save_dir'], 'test.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['FileName', 'Type'])
    for i in range(len(names)):
        name = names[i]
        k = name.rfind('/')
        if k != -1:
            name = name[(k+1):]
        writer.writerow([name, str(int(tot_pred[i])+1)])   
    
print('Csv finished!')
