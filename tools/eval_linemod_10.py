import _init_paths
import argparse
import time
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseNetSym, PoseRefineNet, PoseRefineNetSym
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
    parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
    parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
    parser.add_argument('--model_sym', type=str, default = '',  help='resume PoseNet model for symmetric Objects')
    parser.add_argument('--refine_model_sym', type=str, default = '',  help='resume PoseRefineNet model for symmetric objects')
    opt = parser.parse_args()

    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 600
    iteration = 10 
    bs = 1
    dataset_config_dir = 'datasets/linemod/dataset_config'
    output_result_dir = 'experiments/eval_result/linemod'
    knn = KNearestNeighbor(1)

    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    estimator_sym = PoseNetSym(num_points = num_points, num_obj = num_objects)
    estimator_sym.cuda()

    refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
    refiner.cuda()
    refiner_sym = PoseRefineNetSym(num_points = num_points, num_obj = num_objects)
    refiner_sym.cuda()

    estimator.load_state_dict(torch.load(opt.model))
    estimator_sym.load_state_dict(torch.load(opt.model_sym))
    refiner.load_state_dict(torch.load(opt.refine_model))
    refiner_sym.load_state_dict(torch.load(opt.refine_model_sym))
    estimator_sym.eval()
    estimator.eval()
    refiner.eval()
    refiner_sym.eval()

    testdataset = PoseDataset_linemod('test', num_points, False, opt.dataset_root, 0.0, True)
    print(len(testdataset))
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    print("testdataloader")

    # print(len(testdataset.classes),testdataset.classes)
    sym_list = testdataset.get_sym_list()
    print("symlist = ",sym_list)
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    diameter = []
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.load(meta_file)
    iteration = 10
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')
    # print(testdataloader)

    cnt =0
    st_time = time.time()
    
    for i, data in enumerate(testdataloader,start = 0):
        #cnt+=1
        # if(cnt>500):
        #     break        
        points, choose, img, target, model_points, idx = data
        # print(idx.item())
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            continue
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                        Variable(choose).cuda(), \
                                                        Variable(img).cuda(), \
                                                        Variable(target).cuda(), \
                                                        Variable(model_points).cuda(), \
                                                        Variable(idx).cuda()

        
        if(idx in sym_list and opt.model_sym != ''):
            pred_r, pred_t, pred_c, emb = estimator_sym(img, points, choose, idx)
        else:
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()

            if(idx in sym_list and opt.refine_model_sym!=''):
                pred_r, pred_t = refiner_sym(new_points, emb, idx)
            else:
                pred_r, pred_t = refiner(new_points, emb, idx)


            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()
        
        # print(target.shape," ",pred.shape)

        if idx in sym_list:
            # print("in sym")
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            # print(target.shape," ",pred.shape)
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            # print(inds.shape)
            target = torch.index_select(target, 1, inds.view(-1))
            # print(target.shape," ",pred.shape)
            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0)
            # dis = torch.min(dis,torch.tensor(1))
            # dis = dis.item()
        else:
            # print("not sym")
            dis = np.mean(np.linalg.norm(pred - target, axis=1))

        # print(target.shape," ",pred.shape)
        # print(dis)
        
        if dis < diameter[idx[0]]:#.item()]:
            success_count[idx[0]] += 1 # .item()
            print('No.{0} Pass! Distance: {1} c.m.'.format(i, dis*100))
            fw.write('No.{0} Pass! Distance: {1} c.m.\n'.format(i, dis*100))
        else:
            print('No.{0} NOT Pass! Distance: {1}c.m.'.format(i, dis*100))
            fw.write('No.{0} NOT Pass! Distance: {1}c.m.\n'.format(i, dis*100))
        num_count[idx[0].item()] += 1

    end_time = time.time()
    print(time.gmtime(end_time - st_time))
    print(end_time - st_time)

    for i in range(num_objects):
        print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i])*100 / (num_count[i]+.001)))
        fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / (num_count[i]+0.001)))
    print('ALL success rate: {0}'.format(float(sum(success_count)) / (sum(num_count)+0.001)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / (sum(num_count)+0.001)))
    fw.close()
