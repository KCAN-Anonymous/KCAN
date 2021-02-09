import os
import numpy as np
import scipy.sparse as sp
import math
import heapq
import json
from multiprocessing import Pool
from functools import partial

def choose_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    sort_id = np.argsort(memory_gpu)[::-1]
    gpuid = sort_id[0]
    os.system('rm tmp')
    print('using GPU {}, memory: {}M'.format(gpuid, memory_gpu[gpuid]))
    return gpuid

def make_batch(total_num, batch_size=None, shuffle=True, *args):
    if batch_size is None:
        batch_size = total_num
    n = math.ceil(total_num*1.0 / batch_size)
    if shuffle:
        reId = np.random.permutation(total_num)
    else:
        reId = np.arange(total_num)
    reId = list(reId)
    index = 0
    for i in range(n):
        yield [i, n]+[arg[reId[index:index+batch_size]] for arg in args]
        index += batch_size

class EarlyStop(object):
    def __init__(self, patience=5, save_dir=None):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.num = 0
        self.save_dir = save_dir

    def run(self, loss, model=None):
        score = loss
        if self.best_score is None:
            self.best_score = score
            if self.save_dir is not None:
                self.save_dir.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.save_dir)
        elif score > self.best_score:
            self.num += 1
            if self.num >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.num = 0
            if self.save_dir is not None:
                torch.save(model.state_dict(), self.save_dir)
        return self.early_stop

def test_one_user(score, neg_scores, K):
    s = np.hstack((score, neg_scores))
    ind = heapq.nlargest(K, range(len(s)), s.take)
    for i, m in enumerate(ind):
        if m == 0:
            return 1.0, math.log(2) / math.log(i+2) 
    return 0.0, 0.0

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def save_args(path, args):
    ensureDir(path)
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(path, args):
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)
    return args

def subgraph(adjlist, data, nodes, K):
    nei1 = adjlist[nodes].reshape(len(nodes), -1)
    data1 = data[nodes].reshape(len(nodes), -1)
    nei2 = adjlist[nei1].reshape(len(nodes), -1)
    data2 = data[nei1].reshape(len(nodes), -1)
    return nei1, nei2, data1, data2

def _topk(args, k):
    data, row = args
    data, row = np.array(data), np.array(row)
    if len(data) <= k:
        ind = np.random.choice(range(len(row)), k, p=data/data.sum())
    else:
        #ind = np.argpartition(data, -k)[-k:]
        ind = np.random.choice(range(len(row)), k, replace=False, p=data/data.sum())
    return data[ind], row[ind]

def topk(A, k):
    m = A.tolil()
    my_topk = partial(_topk, k=k)
    with Pool(10) as p:
        res = p.map(my_topk, zip(m.data, m.rows))
    data, row = list(zip(*res))
    data, row = np.array(data), np.array(row)
    return row, data

def softmax(x):
    expx = np.exp(x)
    return expx/expx.sum(0)

if __name__ == '__main__':
    N, M = 5, 10
    row = np.random.randint(0, N, M)
    col = np.random.randint(0, N, M)
    values = np.random.rand(M)
    A = sp.coo_matrix((values, (row, col)), shape=(N, N))
    #A = A+A.T
    print(A)
    res = topk(A, 3)
    print(res)
    #print(subgraph(*res, [0, 1], 2))
