import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from collections import ChainMap
import time
import json

import argparse
import configparser
import tensorflow as tf
import numpy as np

from data_loader import Dataset
from utils import choose_gpu, make_batch, save_args
from EKGCN import EKGCN
from evalution import Model_test

def parse_args():
    parser = argparse.ArgumentParser("Explainable GCN with Knowledge Graph")
    parser.add_argument('--model_type', type=str, default='EKGCN', help='model type.')
    parser.add_argument('--dataset', type=str, default='yelp2018', help='dataset name.')
    parser.add_argument('--result_dir', type=str, default='', help='result directory.')
    parser.add_argument('--config', type=str, default='', help='config file.')

    parser.add_argument('--task', type=str, default='link_prediction', help='task. [link_prediction|node_classification]')
    parser.add_argument('--evalution', type=str, default='topk', help='[topk|ntr|nc]')

    parser.add_argument('-s', '--embedding_size', type=int, default=16, help='the embedding dimension size')
    parser.add_argument('--out_dim', type=int, default=16, help='')
    parser.add_argument('--kg_ui', type=int, default=1, help='')
    parser.add_argument('--kge_size', type=int, default=16, help='the knowledge graph embedding dimension size')
    parser.add_argument('--kge_type', type=str, default='transH', help='')
    parser.add_argument('--max_degree', type=int, default=20, help='max degree per node')
    parser.add_argument('--layer_size', nargs='?', default='[16, 8]', help='Output sizes of every layer')
    parser.add_argument('--reg', type=float, default='0.0001', help='Regularizer')
    parser.add_argument('--dropout', type=float, default='0.1', help='Dropout')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=200, help='Number of epoch to train. Each epoch processes the training data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Number of training examples processed per step')
    parser.add_argument('--kge_batch_size', type=int, default=256, help='Number of training examples processed per step in kge part')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--test_every_iter', type=int, default=100, help='print test result every N iters')
    parser.add_argument('--use_pretrain_kge', type=int, default=1, help='whether to use pretrained knowledge graph embedding')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    if len(args.config) > 0:
        config = configparser.SafeConfigParser()
        config.read(args.config)
        for k, v in config.items('default'):
            parser.parse_args(['--'+str(k), str(v)], args)
    if len(args.result_dir) == 0:
        args.result_dir = 'result/{}/{}'.format(args.dataset, args.model_type)
    return args

def main(args):
    gpuid = choose_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = Dataset(args.dataset, args.task)
    model = EKGCN(args, dataset, sess=sess)
    
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    evalution = Model_test(args.dataset, args.model_type, None, None, sess=sess, model=model)
    
    if args.kg_ui:
        dataset.get_full_kg()

    res = evalution.test(dataset, args.evalution)
    print('Evalution: {}'.format(res))

    for epoch in range(args.epochs_to_train):
        t1 = time.time()
        if args.kg_ui:
            kg = dataset.fkg
        else:
            kg = dataset.kg
        #args.kge_batch_size = None
        loss1, kg_loss, reg_loss1 = 0.0, 0.0, 0.0
        for num, total_num, data in make_batch(dataset.n_triples, args.kge_batch_size, True, kg):
            feed_dict = model.generate_feed_dict_I(data)
            _, batch_loss1, batch_kg_loss, batch_reg_loss1, batch_kg_pi = model.train_I(sess, feed_dict)
            loss1 += batch_loss1
            kg_loss += batch_kg_loss
            reg_loss1 += batch_reg_loss1
            assert not np.isnan(batch_kg_loss)
            if num % args.test_every_iter == 0:
                print("Epoch {}, Phase I, iter {}/{}, loss: {:.4f}, time: {:.2f}s".format(epoch, num, total_num, loss1/(num+1), time.time()-t1))
        t2 = time.time()
        loss2, base_loss, reg_loss = 0.0, 0.0, 0.0
        for num, total_num, data in make_batch(dataset.n_train, args.batch_size, True, dataset.train_edgelist):
            feed_dict = model.generate_feed_dict_II(data)
            _, batch_loss2, batch_base_loss, batch_reg_loss = model.train_II(sess, feed_dict)
            
            loss2 += batch_loss2
            base_loss += batch_base_loss
            reg_loss += batch_reg_loss
            if num % args.test_every_iter == 0:
                print("Epoch {}, Phase II, iter {}/{}, loss: {:.4f}, time: {:.2f}s".format(epoch, num, total_num, loss2/(num+1), time.time()-t2))
        t3 = time.time()
        model.update_A(sess)
        print_info = 'Epoch {}, {:.1f}s: loss=[{:.4f}=={:.4f}+{:.4f}, {:.4f}=={:.4f}+{:.4f}]'.format(epoch, t3-t1, loss1, kg_loss, reg_loss1, loss2, base_loss, reg_loss)
        print(print_info)
        save_args(args.result_dir+'/args.txt', args)
        saver.save(sess, args.result_dir+'/weights', global_step=epoch)
        
        res = evalution.test(dataset, args.evalution)
        print('Evalution: {}'.format(res))


if __name__ == '__main__':
    main(parse_args())
