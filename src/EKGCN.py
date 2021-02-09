import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from pathlib import Path

from evalution import Model_test 
from utils import topk, subgraph

class EKGCN(object):
    def __init__(self, args, dataset, pretrain_data=None, sess=None):
        self._parse_args(args, dataset, pretrain_data, sess)

        self._build_inputs()
        self.weights = self._build_weights()
        self._build_A()
        self.update_A(sess, init=True)
        self._build_model_phase_I()
        self._build_model_phase_II()
        self._statistics_params()

    def _parse_args(self, args, dataset, pretrain_data, sess):
        self.model_type = args.model_type
        print('run {}'.format(self.model_type))
        self.args = args
        self.pretrain_data = pretrain_data
        self.result_path = Path(args.result_dir)
        self.dataset = dataset

        self.A_in = None
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_entities = dataset.n_entities
        self.n_relations = dataset.n_relations

        self.lr = args.learning_rate
        self.layer_size = eval(args.layer_size)

        self.emb_dim = args.embedding_size
        self.batch_size = args.batch_size

        self.kge_dim = args.embedding_size
        self.kge_batch_size = args.kge_batch_size

    def _build_inputs(self):
        self.users = tf.placeholder(tf.int32, shape=(None, ))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.A_values = tf.placeholder(tf.float32, shape=(None,))
        self.A_tensor = tf.sparse_placeholder(tf.float32, shape=[self.n_entities+self.n_users, self.n_entities+self.n_users])

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        self.dropout = tf.placeholder(tf.float32)

        self.adjlist_tensor = tf.placeholder(tf.int32, shape=[None, self.args.max_degree])
        self.datalist_tensor = tf.placeholder(tf.float32, shape=[None, self.args.max_degree])

    def _build_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]), name='entity_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True, name='user_embed', dtype=tf.float32)
            item_embed = self.pretrain_data['item_embed']
            other_embed = initializer([self.n_entities - self.n_items, self.emb_dim])
            all_weights['entity_embed'] = tf.Variable(initial_value=tf.concat([item_embed, other_embed], 0),
                    trainable=True, name='entity_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]), name='relation_embed')
        all_weights['relation_norm'] = tf.Variable(initializer([self.n_relations, self.kge_dim]), name='relation_norm')

        out_dim = self.emb_dim
        if self.model_type in ['EKGCN', 'EKGCN_n']:
            all_weights['gcn_W_0'] = tf.Variable(initializer([2*self.emb_dim, self.layer_size[0]]), name='gcn_W_0')
            all_weights['gcn_b_0'] = tf.Variable(initializer([1, self.layer_size[0]]), name='gcn_b_0')

            out_dim += self.layer_size[0]

        if self.model_type in ['EKGCN', 'EKGCN_g']:
            tn = 2 if self.args.task == 'link_prediction' else 1
            size_in = out_dim
            all_weights['gcn_W_t_1'] = tf.Variable(initializer([tn*size_in, 1]), name='gcn_W_t_1')
            all_weights['gcn_W_e_1'] = tf.Variable(initializer([size_in, 1]), name='gcn_W_t_1')
            all_weights['gcn_W_1'] = tf.Variable(initializer([2*size_in, self.layer_size[1]]), name='gcn_W_1')
            all_weights['gcn_b_1'] = tf.Variable(initializer([1, self.layer_size[1]]), name='gcn_b_1')

            #all_weights['gcn_W_e_2'] = all_weights['gcn_W_e_1']
            #all_weights['gcn_W_2'] = all_weights['gcn_W_1']
            #all_weights['gcn_b_2'] = all_weights['gcn_b_1']
            all_weights['gcn_W_e_2'] = tf.Variable(initializer([self.layer_size[1], 1]), name='gcn_W_t_2')
            all_weights['gcn_W_2'] = tf.Variable(initializer([size_in+self.layer_size[1], self.layer_size[2]]), name='gcn_W_2')
            all_weights['gcn_b_2'] = tf.Variable(initializer([1, self.layer_size[2]]), name='gcn_b_2')
            out_dim += self.layer_size[2]
        all_weights['gcn_output'] = tf.Variable(initializer([out_dim, self.args.out_dim]), name='gcn_output')
        return all_weights

    def _build_model_phase_I(self):
        self.pos_kg_score, self.kg_pi, reg_e = self.get_kg_score(self.h, self.pos_t, self.r, return_att=True)
        self.neg_kg_score = self.get_kg_score(self.h, self.neg_t, self.r)
        self.kg_loss = tf.reduce_mean(tf.nn.softplus(-(self.neg_kg_score - self.pos_kg_score)))
        regularizer = tf.reduce_sum([tf.nn.l2_loss(e) for e in reg_e])
        if self.kge_batch_size is None:
            self.kge_batch_size = self.dataset.n_triples
        self.reg_loss1 = self.args.reg * regularizer / self.kge_batch_size
        self.loss1 = self.kg_loss+self.reg_loss1
        self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss1)

    def _build_model_phase_II(self):
        ea_embeddings, ua_embeddings = self.weights['entity_embed'], self.weights['user_embed']
        if self.model_type in ['EKGCN_n', 'EKGCN']:
            ea_embeddings, ua_embeddings = self.graphsage_layer(ea_embeddings, ua_embeddings, self.A_tensor)
            #ea_embeddings, ua_embeddings = self.graphsage_layer(ea_embeddings, ua_embeddings, self.spmat2tensor(self.A_in))
        if self.model_type in ['EKGCN_n', 'EKGCN_s']:
            self.u_e = tf.nn.embedding_lookup(ua_embeddings, self.users)
            self.pos_i_e = tf.nn.embedding_lookup(ea_embeddings, self.pos_items)
            self.pos_score = tf.reduce_sum(self.u_e*self.pos_i_e, axis=-1)
            self.neg_i_e = tf.nn.embedding_lookup(ea_embeddings, self.neg_items)
            self.neg_score = tf.reduce_sum(self.u_e*self.neg_i_e, axis=-1) 
            self.pos_score_e = self.pos_score
            self.neg_score_e = self.neg_score
            reg_e = [self.u_e, self.pos_i_e, self.neg_score_e]
        elif self.model_type in ['EKGCN_g', 'EKGCN']:
            self.pos_score_e, self.neg_score_e, reg_e = self.subgraph_agg_layer(ea_embeddings, ua_embeddings)

        reg_e += [v for k, v in self.weights.items() if k.startswith('gcn')]
        regularizer = tf.reduce_sum([tf.nn.l2_loss(e) for e in reg_e])
        self.reg_loss = self.args.reg * regularizer / self.batch_size

        self.base_loss = tf.reduce_mean(tf.nn.softplus(-(self.pos_score_e-self.neg_score_e)))
        self.loss2 = self.base_loss+self.reg_loss
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _build_A(self, undirected=True):
        A_indices = self.dataset.generate_edgeindex()
        if undirected:
            temp = np.vstack((A_indices[:, 1], A_indices[:, 0])).T
            A_indices = np.vstack((A_indices, temp))
            self.A = tf.SparseTensor(A_indices, tf.concat([self.A_values, self.A_values], 0), (self.dataset.n_nodes, self.dataset.n_nodes))
        else:
            self.A = tf.SparseTensor(A_indices, self.A_values, (self.dataset.n_nodes, self.dataset.n_nodes))
        self.A_norm = tf.sparse.softmax(tf.sparse.reorder(self.A))

    def graphsage_layer(self, ea_embeddings, ua_embeddings, A):
        pre_embeddings = tf.concat([ea_embeddings, ua_embeddings], axis=0)
        all_embeddings = [pre_embeddings]

        embeddings = tf.sparse_tensor_dense_matmul(A, pre_embeddings)
        embeddings = tf.concat([pre_embeddings, embeddings], 1)
        #embeddings += pre_embeddings
        pre_embeddings = tf.nn.relu(tf.matmul(embeddings, self.weights['gcn_W_%d' % 0]) + self.weights['gcn_b_%d' % 0])
        pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.dropout)
        norm_embeddings = tf.math.l2_normalize(pre_embeddings, axis=1)
        all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        ea_embeddings, ua_embeddings = tf.split(all_embeddings, [self.n_entities, self.n_users], 0)
        return ea_embeddings, ua_embeddings

    def generate_k_subgraph(self, embeddings, nodes, return_node=False):
        e = tf.nn.embedding_lookup(embeddings, nodes)
        n1 = tf.nn.embedding_lookup(self.adjlist_tensor, nodes)
        ne1 = tf.nn.embedding_lookup(embeddings, n1)
        nd1 = tf.nn.embedding_lookup(self.datalist_tensor, nodes)
        n2 = tf.reshape(tf.nn.embedding_lookup(self.adjlist_tensor, n1), (-1, self.args.max_degree**2))
        ne2 = tf.nn.embedding_lookup(embeddings, n2)
        nd2 = tf.reshape(tf.nn.embedding_lookup(self.datalist_tensor, n1), (-1, self.args.max_degree**2))
        if return_node:
            return (e, ne1, ne2, nd1, nd2), n1, n2
        return e, ne1, ne2, nd1, nd2

    def subgraph_agg_layer(self, ea_embeddings, ua_embeddings):
        pre_embeddings = tf.concat([ea_embeddings, ua_embeddings], axis=0)

        user_info, n1, n2 = self.generate_k_subgraph(pre_embeddings, self.users+self.n_entities, return_node=True)
        self.n1, self.n2 = n1, n2

        pos_items_info, nn1, nn2 = self.generate_k_subgraph(pre_embeddings, self.pos_items, return_node=True)
        self.nn1, self.nn2 = nn1, nn2

        neg_items_info = self.generate_k_subgraph(pre_embeddings, self.neg_items)

        reg_e = []

        targets = tf.concat([user_info[0], pos_items_info[0]], axis=-1)
        u_e, atts1 = self.subgraph_agg(targets, *user_info, return_att=True)
        self.atts1 = atts1
        pi_e, atts2 = self.subgraph_agg(targets, *pos_items_info, return_att=True)
        self.atts2 = atts2
        self.pos_score_e = tf.reduce_sum(u_e*pi_e, axis=-1)
        reg_e += [u_e, pi_e]

        targets = tf.concat([user_info[0], neg_items_info[0]], axis=-1)
        u_e = self.subgraph_agg(targets, *user_info)
        ni_e = self.subgraph_agg(targets, *neg_items_info)
        self.neg_score_e = tf.reduce_sum(u_e*ni_e, axis=-1)
        reg_e += [u_e, ni_e]
        return self.pos_score_e, self.neg_score_e, reg_e

    def subgraph_agg(self, targets, ue, ue1, ue2, ud1, ud2, use_knowledge_weight=True, return_att=False):
        atts = []
        all_embeddings = [ue]
        tv = tf.matmul(targets, self.weights['gcn_W_t_1']) #b*1
        ev = tf.squeeze(tf.tensordot(ue2, self.weights['gcn_W_e_1'], axes=1), axis=-1) #b*K^2
        att = tv+ev
        if use_knowledge_weight:
            att = tf.multiply(ud2, att)
        att = tf.nn.softmax(tf.reshape(tf.nn.leaky_relu(att), [-1, self.args.max_degree, self.args.max_degree]), axis=-1) #b*K*K
        atts.append(att)
        new_ue1 = tf.reduce_sum(tf.multiply(tf.expand_dims(att, -1), tf.reshape(ue2, [-1, self.args.max_degree, self.args.max_degree, ue2.shape[-1]])), axis=-2) #b*K*d
        ue1 = tf.concat([ue1, new_ue1], axis=-1)
        ue1 = tf.tensordot(ue1, self.weights['gcn_W_%d' % 1], axes=1) + self.weights['gcn_b_%d' % 1]
        ue1 = tf.nn.elu(ue1)
        ue1 = tf.nn.dropout(ue1, 1-self.dropout)
        ue1 = tf.math.l2_normalize(ue1, axis=-1) #b*K*dd

        ev = tf.squeeze(tf.tensordot(ue1, self.weights['gcn_W_e_2'], axes=1), axis=-1) #b*K 
        att = tv+ev
        if use_knowledge_weight:
            att = tf.multiply(ud1, att)
        att = tf.nn.softmax(tf.nn.leaky_relu(att), axis=-1)
        atts.append(att)
        new_ue = tf.reduce_sum(tf.multiply(tf.expand_dims(att, -1), ue1), axis=-2) #b*dd
        #new_ue = tf.reduce_mean(ue1, axis=-2)
        ue = tf.concat([ue, new_ue], axis=-1)
        ue = tf.matmul(ue, self.weights['gcn_W_%d' % 2])+self.weights['gcn_b_%d' % 2]
        ue = tf.nn.elu(ue)
        ue = tf.nn.dropout(ue, 1-self.dropout)
        ue = tf.math.l2_normalize(ue, axis=-1)
        all_embeddings += [ue]
        all_embeddings = tf.concat(all_embeddings, axis=-1)
        all_embeddings = tf.matmul(all_embeddings, self.weights['gcn_output'])
        if return_att:
            return all_embeddings, atts
        return all_embeddings

    def get_kg_score(self, h, t, r, return_att=False):
        embeddings = tf.concat([self.weights['entity_embed'], self.weights['user_embed']], axis=0)
        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)
        ### Warning: on some version in gpu, the out-of-bound key will have all zero values
        r_e = tf.nn.embedding_lookup(tf.concat([self.weights['relation_embed'], tf.zeros([1, self.weights['relation_embed'].shape[1]])], 0), r)
        r_n = tf.nn.embedding_lookup(tf.concat([self.weights['relation_norm'], tf.zeros([1, self.weights['relation_norm'].shape[1]])], 0), r)
        h_e = self.kg_trans(h_e, r_n)
        t_e = self.kg_trans(t_e, r_n)

        h_e = tf.math.l2_normalize(h_e, axis=-1)
        t_e = tf.math.l2_normalize(t_e, axis=-1)
        r_e = tf.math.l2_normalize(r_e, axis=-1)

        reg_e = [h_e, t_e, r_e, tf.nn.l2_normalize(r_n)]

        score = tf.norm(h_e+r_e-t_e, ord=1, axis=-1)
        if return_att:
            att = tf.reduce_sum(tf.multiply(t_e, h_e + r_e), 1)
            return score, att, reg_e 
        return score

    def kg_trans(self, e, norm):
        norm = tf.math.l2_normalize(norm, axis=-1)
        return e-tf.multiply(tf.reduce_sum(tf.multiply(norm, e), 1, keepdims=True), norm)

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

    def train_I(self, sess, feed_dict):
        return sess.run([self.opt1, self.loss1, self.kg_loss, self.reg_loss1, self.kg_pi], feed_dict)
    
    def train_II(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.base_loss, self.reg_loss], feed_dict)

    def spmat2tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def sptensor2mat(self, X):
        return sp.coo_matrix((X.values, (X.indices[:, 0], X.indices[:, 1])), shape=X.dense_shape)

    def update_A(self, sess, init=False):
        if init:
            A_values = np.ones(self.dataset.kg.shape[0]+self.dataset.train_edgelist.shape[0])
        else:
            feed_dict = {
                self.h: self.dataset.kg[:, 0],
                self.r: self.dataset.kg[:, 1],
                self.pos_t: self.dataset.kg[:, 2],
                self.users: self.dataset.train_edgelist[:, 0],
                self.pos_items: self.dataset.train_edgelist[:, 1],
                self.dropout: 0.0,
                }
            if self.args.kg_ui:
                feed_dict = {
                    self.h: self.dataset.fkg[:, 0],
                    self.r: self.dataset.fkg[:, 1],
                    self.pos_t: self.dataset.fkg[:, 2],
                    self.dropout: 0.0,
                    }
                A_values = np.hstack(sess.run(self.kg_pi, feed_dict=feed_dict))
            else:
                A_values = np.hstack(sess.run([self.kg_pi, self.pos_score_e], feed_dict=feed_dict))
        self.a_tensor = sess.run(self.A_norm, feed_dict={self.A_values: A_values})
        self.A_in = self.sptensor2mat(self.a_tensor)
        if self.model_type in ['EKGCN', 'EKGCN_g']:
            self.adjlist, self.datalist = topk(self.A_in, self.args.max_degree)

    def generate_feed_dict_I(self, data):
        feed_dict = {
                self.h: data[:, 0],
                self.r: data[:, 1],
                self.pos_t: data[:, 2],
                self.neg_t: self.dataset.negative_sample_kg(data),
                }
        return feed_dict

    def generate_feed_dict_II(self, data, with_neg=True):
        users = data[:, 0]
        pos_items = data[:, 1]
        feed_dict = {
                self.users: data[:, 0],
                self.pos_items: data[:, 1],
                self.dropout: self.args.dropout,
                self.A_tensor: self.a_tensor,
                }
        if with_neg:
            neg_items = self.dataset.negative_sample(data)
            feed_dict[self.neg_items] = neg_items
        if self.model_type in ['EKGCN', 'EKGCN_g']:
            dict_temp = {
                self.adjlist_tensor: self.adjlist,
                self.datalist_tensor: self.datalist,
                }
            feed_dict.update(dict_temp)
        return feed_dict
