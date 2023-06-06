import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
# import knn_pytorch.knn as knn_pytorch
from ctypes import *


class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  def __init__(self, k):
    self.k = k

  def knn_1(self,ref, query, k = 1):
    # print(ref.transpose(-1,0))
    # print(query)
    mat = torch.zeros(ref.shape[2],query.shape[2]).cuda()

    for i in range(0,ref.shape[0]):

      for idx in range(0,ref.shape[1]):
        ref1 = ref[i,idx,:].unsqueeze(0).transpose(-1,0)@torch.ones(ref.shape[2]).unsqueeze(0).cuda()
        query1 = torch.transpose(query[i,idx,:].unsqueeze(0).transpose(-1,0)@torch.ones(query.shape[2]).unsqueeze(0).cuda(),-1,0)

        t1 = ref1 - query1
        # print(torch.norm(ref[i,idx,:].transpose(-1,0) - query[i,idx,:].transpose(-1,0)))
        mat+=t1.abs()
    # mat = torch.sqrt(mat)/ref.shape[1]
    # print(torch.ones(ref.shape[2]).unsqueeze(0).transpose(-1,0))
    # print(ref[0,0,:].unsqueeze(0))
    # print(ref[0,0,:].unsqueeze(0).transpose(-1,0)@torch.ones(ref.shape[2]).unsqueeze(0).cuda())
    # print(torch.transpose(query[0,0,:].unsqueeze(0).transpose(-1,0)@torch.ones(query.shape[2]).unsqueeze(0).cuda(),-1,0))
    sorted,index = torch.sort(mat,1,False)
    # print(sorted)
    # print(sorted[:, :k])
    return index[:, :k]

    

  def knn1(self,ref, query, k=1):
    print("q ",query.shape)
    print("r ",ref.shape)
    mat = torch.Tensor(ref.shape[2],query.shape[2])

    for i in range(0,ref.shape[2]):
      for j in range(0,query.shape[2]):
        t = 0
        for idx in range(0,ref.shape[1]):
          t += torch.pow(query[0,idx,j] - ref[0,idx,i],2)
        mat[i,j] = torch.sqrt(t)
    
    sorted,indices = torch.sort(mat,0)

    return indices[:, :]

  def forward(self, ref, query,k=1):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = self.knn_1(ref,query,1)
    inds = inds.long().cuda()
    
    return inds

    def backward(self, ref, query,k=1):
      ref = ref.float().cuda()
      query = query.float().cuda()

      inds = self.knn_1(ref,query,1)
      inds = inds.long().cuda()
      
      return inds



class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    knn = KNearestNeighbor(1)
    cnt=0
    while(cnt<1):
        cnt+=1
        D, N, M = 1, 2, 2
        ref = Variable(torch.rand(1, D, int(N)))
        query = Variable(torch.rand(1, D, int(M)))

        inds = knn(ref, query)
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        ref = ref.cpu()
        query = query.cpu()
        print(inds)


if __name__ == '__main__':
  unittest.main()
