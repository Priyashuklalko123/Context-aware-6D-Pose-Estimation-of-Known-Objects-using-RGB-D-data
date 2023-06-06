import os
import random
import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

from data_controller_linemod import PoseDataset as PoseDataset_linemod
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='../datasets/linemod/Linemod_preprocessed')
parser.add_argument('--batch_size', default=3, help="batch size")
parser.add_argument('--n_epochs', default=100, help="epochs to train")
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/linemod/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/linemod/', help="path to save models")
parser.add_argument('--log_dir', default='logs/linemod/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument('--dataset', type=str, default = 'linemod')
parser.add_argument('--noise_trans', default=0.03)
opt = parser.parse_args()

if __name__ == '__main__':
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.repeat_epoch = 20
        opt.refine_start = False
    else:
        print('Unknown dataset')

    if opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    
    print(len(dataset), len(test_dataset))

    model = segnet()
    model = model.cuda()

    if opt.resume_model != '':
        checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
        model.load_state_dict(checkpoint)
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = Loss()
    best_val_cost = np.Inf
    st_time = time.time()

    for epoch in range(1, opt.n_epochs):
        model.train()
        train_all_cost = 0.0
        train_time = 0
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))

        for i, data in enumerate(dataloader, 0):
            rgb, target = data
            # print(rgb.shape, target.shape)
            rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
            semantic = model(rgb)
            # print(semantic.shape)

            optimizer.zero_grad()
            
            semantic_loss = criterion(semantic, target)
            train_all_cost += semantic_loss.item()
            semantic_loss.backward()
            optimizer.step()
            logger.info('Train time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), train_time, semantic_loss.item()))
            if train_time != 0 and train_time % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_current.pth'))
            train_time += 1

        train_all_cost = train_all_cost / train_time
        logger.info('Train Finish Avg CEloss: {0}'.format(train_all_cost))
        
        model.eval()
        test_all_cost = 0.0
        test_time = 0
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        for j, data in enumerate(testdataloader, 0):
            rgb, target = data
            rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
            semantic = model(rgb)
            semantic_loss = criterion(semantic, target)
            test_all_cost += semantic_loss.item()
            test_time += 1
            logger.info('Test time {0} Batch {1} CEloss {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_time, semantic_loss.item()))

        test_all_cost = test_all_cost / test_time
        logger.info('Test Finish Avg CEloss: {0}'.format(test_all_cost))

        if test_all_cost <= best_val_cost:
            best_val_cost = test_all_cost
            torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'model_{}_{}.pth'.format(epoch, test_all_cost)))
            print('----------->BEST SAVED<-----------')
