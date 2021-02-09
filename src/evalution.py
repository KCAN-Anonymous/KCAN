import os, sys
from pathlib import Path

import configparser
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score

from data_loader import Dataset
from utils import choose_gpu, make_batch, test_one_user, load_args, softmax

KGAT_PATH = '/home/tuke/knowledge/knowledge_graph_attention_network/Model/'
EKGCN_TORCH_PATH = '/home/tuke/knowledge/EKGCN/result'
#EKGCN_PATH = '/home/tuke/knowledge/EKGCN_tf/result'
EKGCN_PATH = '/Users/tuke/Documents/study/knowledge_and_reasoning/code/knowledge/EKGCN_tf/result'
RIPPLE_PATH = '/home/tuke/knowledge/RippleNet/src'

class Model_test(object):
    def __init__(self, dataset_name, model_name, model_dir, arg_file, **argc):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_dir = model_dir
        self.arg_file = arg_file
        if 'model' in argc:
            self.model = argc['model']
            self.sess = argc['sess']
        else:
            if model_name == 'kgat':
                sys.path.append(KGAT_PATH)
                from KGAT import KGAT
                from utility.loader_kgat import KGAT_loader

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)

                self.args = self.build_args()
                data = KGAT_loader(args=self.args, path='data/{}'.format(dataset_name))
                config = self.build_config(data)

                self.model = KGAT(data_config=config, pretrain_data=None, args=self.args)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.model.update_attentive_A(self.sess)
            elif model_name == 'cke':
                sys.path.append(KGAT_PATH)
                from CKE import CKE
                from utility.loader_cke import CKE_loader

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)

                self.args = self.build_args()
                data = CKE_loader(args=self.args, path='data/{}'.format(dataset_name))
                config = self.build_config(data)

                self.model = CKE(data_config=config, pretrain_data=None, args=self.args)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
            elif model_name == 'ripple':
                sys.path.append(RIPPLE_PATH)
                from ripple_model import RippleNet
                from ripple_data_loader import load_data as ld

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)

                self.args = self.build_args()
                self.args.dataset = dataset_name
                self.loader = ld(self.args)
                
                data = argc['data']
                self.model = RippleNet(self.args, data.n_entities, data.n_relations)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    saver.restore(self.sess, ckpt.model_checkpoint_path)

            elif model_name == 'cfkg':
                sys.path.append(KGAT_PATH)
                from CFKG import CFKG
                from utility.loader_cfkg import CFKG_loader

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)

                self.args = self.build_args()
                data = CFKG_loader(args=self.args, path='data/{}'.format(dataset_name))
                config = self.build_config(data)

                self.model = CFKG(data_config=config, pretrain_data=None, args=self.args)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
            elif model_name == 'nfm':
                sys.path.append(KGAT_PATH)
                from NFM import NFM
                from utility.loader_nfm import NFM_loader

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)

                self.args = self.build_args()
                data = NFM_loader(args=self.args, path='data/{}'.format(dataset_name))
                self.loader = data
                config = self.build_config(data)

                self.model = NFM(data_config=config, pretrain_data=None, args=self.args)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
            elif model_name == 'EKGCN_torch':
                from model import Model
                self.device = torch.device('cuda:{}'.format(argc['gpu_id']))
                self.model = Model.load_checkpoint(self.model_dir / 'model.pt', self.device).to(self.device)
                self.user_score = {} #cached
            elif model_name in ['EKGCN_s', 'EKGCN_g', 'EKGCN_n', 'EKGCN']:
                from EKGCN import EKGCN
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.args = self.build_args()
                data = argc['data']
                self.model = EKGCN(self.args, data, sess=self.sess)
                data.get_full_kg()
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_dir / 'checkpoint'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.sess.run(tf.global_variables_initializer())
                    print('>>> restore from {}'.format(ckpt.model_checkpoint_path))
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.model.update_A(self.sess)

    def build_args(self):
        if self.model_name in ['kgat', 'cfkg', 'nfm', 'cke']:
            from utility.parser import parse_args
        elif self.model_name.startswith('EKGCN'):
            from main import parse_args
        elif self.model_name == 'ripple':
            from ripple_main import parse_args
        args = parse_args()
        if self.arg_file is not None:
            args = load_args(self.arg_file, args)
        return args

    def build_config(self, data):
        config = {}
        config['n_users'] = data.n_users
        config['n_items'] = data.n_items
        config['n_relations'] = data.n_relations
        config['n_entities'] = data.n_entities
        if self.model_name in ['kgat', 'cfkg']:
            config['A_in'] = sum(data.lap_list)
            config['all_h_list'] = data.all_h_list
            config['all_r_list'] = data.all_r_list
            config['all_t_list'] = data.all_t_list
            config['all_v_list'] = data.all_v_list
        return config

    def eval(self, edges, user_shift=0, **argv):
        if self.model_name in ['kgat']:
            feed_dict = {self.model.users: edges[:, 0], 
                        self.model.pos_items: edges[:, 1],
                        self.model.mess_dropout: [0.]*len(eval(self.args.layer_size)),
                        self.model.node_dropout: [0.]*len(eval(self.args.layer_size)),
                        }
            u_e, i_e = self.sess.run([self.model.u_e, self.model.pos_i_e], feed_dict=feed_dict)
            score = np.sum(u_e*i_e, axis=-1)
        elif self.model_name in ['cke']:
            feed_dict = {self.model.u: edges[:, 0],
                        self.model.pos_i: edges[:, 1],
                    }
            u_e, i_e = self.sess.run([self.model.u_e, self.model.pos_i_e], feed_dict=feed_dict)
            score = np.sum(u_e*i_e, axis=-1)
        elif self.model_name == 'ripple':
            users = edges[:, 0]
            feed_dict = {
                    self.model.items: edges[:, 1],
                    }
            for i in range(self.args.n_hop):
                feed_dict[self.model.memories_h[i]] = [self.loader[5][user][i][0] for user in users]
                feed_dict[self.model.memories_r[i]] = [self.loader[5][user][i][1] for user in users]
                feed_dict[self.model.memories_t[i]] = [self.loader[5][user][i][2] for user in users]
            score = self.sess.run(self.model.scores_normalized, feed_dict=feed_dict)
        elif self.model_name == 'cfkg':
            feed_dict = {self.model.h: edges[:, 0],
                        self.model.r: [0]*edges.shape[0],
                        self.model.pos_t: edges[:, 1]+argv['n_users'],
                        self.model.mess_dropout: [0.]*len(eval(self.args.layer_size)),
                        self.model.node_dropout: [0.]*len(eval(self.args.layer_size)),
                    }
            score = self.sess.run(self.model.batch_predictions, feed_dict=feed_dict)
        elif self.model_name == 'nfm':
            user_list = edges[:, 0].tolist()
            item_list = list(edges[:, 1])
            u_sp = self.loader.user_one_hot[user_list]
            pos_i_sp = self.loader.kg_feat_mat[item_list]
            pos_feats = sp.hstack([u_sp, pos_i_sp])
            pos_indices, pos_values, pos_shape = self.loader._extract_sp_info(pos_feats)
            feed_dict = {
                        self.model.pos_indices: pos_indices,
                        self.model.pos_values: pos_values,
                        self.model.pos_shape: pos_shape,
                        self.model.mess_dropout: [0.] * len(eval(self.args.layer_size))
                        }
            score = self.sess.run(self.model.batch_predictions, feed_dict)
        elif self.model_name == 'EKGCN_torch':
            edges[:, 0] += user_shift
            targets = torch.LongTensor(edges).to(self.device)
            score = self.model.forward(targets).detach().cpu().numpy()
        elif self.model_name in ['EKGCN_n', 'EKGCN_g', 'EKGCN_s', 'EKGCN']:
            feed_dict = self.model.generate_feed_dict_II(edges, with_neg=False)
            feed_dict[self.model.dropout] = 0.0
            score = self.sess.run(self.model.pos_score_e, feed_dict=feed_dict)
        return score
    
    def eval2(self, users, item_lists, **argv):
        if self.model_name in ['kgat']:
            feed_dict = {
                        self.model.users: users,
                        self.model.pos_items: item_lists.reshape(-1),
                        self.model.mess_dropout: [0.]*len(eval(self.args.layer_size)),
                        self.model.node_dropout: [0.]*len(eval(self.args.layer_size)),
                        }
            u_e, i_es = self.sess.run([self.model.u_e, self.model.pos_i_e], feed_dict=feed_dict)
            score = np.sum(
                        np.expand_dims(u_e, 1)*\
                        i_es.reshape(item_lists.shape[0], item_lists.shape[1], -1),
                    axis=-1)
        elif self.model_name in ['cke']:
            feed_dict = {self.model.u: users,
                        self.model.pos_i: item_lists.reshape(-1),
                    }
            u_e, i_es = self.sess.run([self.model.u_e, self.model.pos_i_e], feed_dict=feed_dict)
            score = np.sum(
                        np.expand_dims(u_e, 1)*\
                        i_es.reshape(item_lists.shape[0], item_lists.shape[1], -1),
                    axis=-1)
        elif self.model_name == 'ripple':
            users = np.repeat(users, item_lists.shape[1])
            feed_dict = {
                    self.model.items: item_lists.reshape(-1),
                    }
            for i in range(self.args.n_hop):
                feed_dict[self.model.memories_h[i]] = [self.loader[5][user][i][0] for user in users]
                feed_dict[self.model.memories_r[i]] = [self.loader[5][user][i][1] for user in users]
                feed_dict[self.model.memories_t[i]] = [self.loader[5][user][i][2] for user in users]
            score = self.sess.run(self.model.scores_normalized, feed_dict=feed_dict)
            score = score.reshape(item_lists.shape)
        elif self.model_name == 'cfkg':
            feed_dict = {self.model.h: np.repeat(users, item_lists.shape[1]),
                        self.model.r: [0]*item_lists.shape[0]*item_lists.shape[1],
                        self.model.pos_t: item_lists.reshape(-1)+argv['n_users'],
                        self.model.mess_dropout: [0.]*len(eval(self.args.layer_size)),
                        self.model.node_dropout: [0.]*len(eval(self.args.layer_size)),
                    }
            score = self.sess.run(self.model.batch_predictions, feed_dict=feed_dict)
            score = score.reshape(item_lists.shape)
        elif self.model_name == 'nfm':
            user_list = np.repeat(users, item_lists.shape[1]).tolist()
            item_list = item_lists.reshape(-1)
            u_sp = self.loader.user_one_hot[user_list]
            pos_i_sp = self.loader.kg_feat_mat[item_list]
            pos_feats = sp.hstack([u_sp, pos_i_sp])
            pos_indices, pos_values, pos_shape = self.loader._extract_sp_info(pos_feats)
            feed_dict = {
                        self.model.pos_indices: pos_indices,
                        self.model.pos_values: pos_values,
                        self.model.pos_shape: pos_shape,
                        self.model.mess_dropout: [0.] * len(eval(self.args.layer_size))
                        }
            score = self.sess.run(self.model.batch_predictions, feed_dict)
            score = score.reshape(item_lists.shape)
        elif self.model_name == 'EKGCN_torch':
            score = []
            for i in range(users.shape[0]):
                if users[i] not in self.user_score:
                    items = item_lists[i]
                    ts = torch.LongTensor(np.vstack((np.full(items.shape[0], u), items)).T).to(self.device)
                    user_score[users[i]] = self.model.forward(ts).cpu().detach().numpy()
                score.append(user_score[users[i]])
            score = np.vstack(score)
        elif self.model_name in ['EKGCN_n', 'EKGCN_s', 'EKGCN_g', 'EKGCN']:
            edges = np.vstack((np.repeat(users, item_lists.shape[-1]), item_lists.reshape(-1))).T
            feed_dict = self.model.generate_feed_dict_II(edges, with_neg=False)
            feed_dict[self.model.dropout] = 0.0
            score = self.sess.run(self.model.pos_score_e, feed_dict=feed_dict).reshape(item_lists.shape)
        return score

    def test(self, data, task):
        print('test {} {} {}'.format(self.dataset_name, task, self.model_name))
        res = []
        if task in ['topk', 'all']:
            K = 10
            n = 10000
            batch_size = 64
            n_users = data.n_users
            n_items = data.n_items
            ind = np.random.choice(data.n_test, size=n, replace=False)
            data.init_test()
            edgelist = data.test_edgelist[ind]
            user_score = {}
            hrs, ndcgs = [], []
            for num, total_num, edges in make_batch(n, batch_size, True, edgelist):
                score = self.eval(edges, user_shift=data.n_entities, n_users=n_users)
                users = edges[:, 0]
                item_lists = data.test_negative[users]
                score_neg = self.eval2(users, item_lists, n_users=n_users)
                for pos, neg in zip(score, score_neg):
                    hit_ratio, ndcg = test_one_user(pos, neg, K)
                    hrs.append(hit_ratio)
                    ndcgs.append(ndcg)
            res += [np.mean(hrs), np.mean(ndcgs)]
        if task in ['ctr', 'all']:
            n = 10000
            batch_size = 1000
            n_test = data.n_test
            ind = np.random.choice(data.n_test, size=n, replace=False)
            edgelist = data.test_edgelist[ind]
            aucs, f1s = [], []
            for num, total_num, edges, in make_batch(n, batch_size, True, edgelist):
                neg_items = data.negative_sample(edges) 
                edges = np.vstack((edges, np.vstack((edges[:, 0], neg_items)).T))
                y = np.hstack((np.ones(neg_items.shape[0]), np.zeros(neg_items.shape[0])))
                score = softmax(self.eval(edges, user_shift=data.n_entities, n_users=data.n_users))
                auc = roc_auc_score(y_true=y, y_score=score)
                #score[score >= 0.5] = 1
                #score[score < 0.5] = 0
                #f1 = f1_score(y_true=y, y_pred=score)
                aucs.append(auc)
                #f1s.append(f1)
            #res = [np.mean(aucs), np.mean(f1s)]
            res += [np.mean(aucs)]

        return res

if __name__ == '__main__':
    dataset_name = 'last-fm'
    task = 'all'
    model_name = 'EKGCN'
    
    infos = {'kgat': {'model_name': 'kgat',
                #'model_dir' : Path(KGAT_PATH) / 'weights/{}/kgat_si_sum_bi_l3/16-8-8/l0.0001_r1e-05-1e-05'.format(dataset_name),
                'model_dir' : Path(KGAT_PATH) / 'weights/{}/kgat_si_sum_bi_l1/8/l0.0001_r1e-05-1e-05'.format(dataset_name),
                'arg_file': 'args.txt'},
            'cke': {'model_name': 'cke',
                'model_dir' : Path(KGAT_PATH) / 'weights/{}/cke/l0.0001_r1e-05-1e-05'.format(dataset_name),
                'arg_file': 'args.txt'},
            'ripple': {'model_name': 'ripple',
                'model_dir' : Path(RIPPLE_PATH) / 'weights/{}'.format(dataset_name),
                'arg_file': None},
            'cfkg': {'model_name': 'cfkg',
                'model_dir' : Path(KGAT_PATH) / 'weights/{}/cfkg_si_sum_bi_l3/l0.0001_r1e-05-1e-05'.format(dataset_name),
                'arg_file': 'args.txt'},
            'nfm': {'model_name': 'nfm',
                'model_dir': Path(KGAT_PATH) / 'weights/{}/nfm_l3/16-8-8/l0.0001_r1e-05-1e-05'.format(dataset_name),
                'arg_file': 'args.txt'},
            'EKGCN_torch': {'model_name': 'EKGCN_torch',
                'model_dir' : Path(EKGCN_TORCH_PATH) / dataset_name,
                'arg_file': ''},
            'EKGCN_s': {'model_name': 'EKGCN_s',
                'model_dir': Path(EKGCN_PATH) / '{}/EKGCN_s'.format(dataset_name),
                'arg_file': 'args.txt'},
            'EKGCN_n': {'model_name': 'EKGCN_n',
                'model_dir': Path(EKGCN_PATH) / '{}/EKGCN_n'.format(dataset_name),
                'arg_file': 'args.txt'},
            'EKGCN': {'model_name': 'EKGCN',
                'model_dir': Path(EKGCN_PATH) / '{}/EKGCN'.format(dataset_name),
                'arg_file': 'args.txt'}}

    info = infos[model_name]

    model_name = info['model_name']
    model_dir = info['model_dir']
    if info['arg_file'] is None:
        arg_file = None
    else:
        arg_file = model_dir / info['arg_file']

    #gpu_id = choose_gpu()
    gpu_id = None
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    data = Dataset(dataset_name, 'link_prediction')
    model_test = Model_test(dataset_name, model_name, model_dir, arg_file, gpu_id=gpu_id, data=data)
    print(model_test.test(data, task))
