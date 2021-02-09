from pathlib import Path
import collections
import numpy as np

class Dataset(object):
    def __init__(self, name, task='link_prediction'):
        """
        link prediction/recommendation:
            - train.npy: train edgelist, user-item(-rating)
            - test.npy: test edgelist, user-item(-rating)
            * test_negative.npy: negtive samples for topk test
        node classification:
            - edgelist.npy: train edgelist, user-item
            - train_labels.npy: train, node_id-label
            - test_labels.npy: test, node_id-label
            * groundth.pkl: ground truth for synthetic data 
        - kg_final.npy: knowldge graph, head-relation-tail
        """
        self.name = name
        self.task = task
        self.data_dir = Path('data') / name 
        if task == 'link_prediction':
            self.train_edgelist = self._load_array(self.data_dir, 'train')
            self.test_edgelist = self._load_array(self.data_dir, 'test')

            self.n_users = max(max(self.train_edgelist[:, 0]), max(self.test_edgelist[:, 0])) + 1
            self.n_items = max(max(self.train_edgelist[:, 1]), max(self.test_edgelist[:, 1])) + 1
            self.n_train = len(self.train_edgelist)
            self.n_test = len(self.test_edgelist)
        elif task == 'node_classification':
            self.train_edgelist = self._load_array(self.data_dir, 'edgelist')
            self.n_users = max(self.train_edgelist[:, 0])+1
            self.n_items = max(self.train_edgelist[:, 1])+1
            self.train_labels = self._load_array(self.data_dir, 'train_labels') 
            self.test_labels = self._load_array(self.data_dir, 'test_labels') 
            self.n_train = len(self.train_labels)
            self.n_test = len(self.test_labels)

        print("n_users: {}, n_items: {}, n_train: {}, n_test: {}".format(self.n_users, self.n_items, self.n_train, self.n_test))

        self.kg = self._load_kg(self.data_dir, 'kg_final')
        self.n_relations = max(self.kg[:, 1])+1
        self.n_entities = max(max(self.kg[:, 0]), max(self.kg[:, 2]))+1
        self.n_triples = len(self.kg)
        print("n_relations: {}, n_entities: {}, n_triples: {}".format(self.n_relations, self.n_entities, self.n_triples))

        self.n_nodes = self.n_entities+self.n_users

        self.kg_id, self.train_id = None, None

    def init_test(self):
        self.test_negative = self._load_test_negative(self.data_dir, 'test_negative')

    def test(self):
        assert self.n_users == len(set(self.train_edgelist[:, 0])|set(self.test_edgelist[:, 0]))
        assert self.n_items == len(set(self.train_edgelist[:, 1])|set(self.test_edgelist[:, 1]))
        assert self.n_relations == len(set(self.kg[:, 1]))
        assert self.n_entities == len(set(self.kg[:, 0])| set(self.kg[:, 2]))

    def _load_array(self, fpath, name):
        npy_file = fpath / "{}.npy".format(name)
        edges = np.load(npy_file)
        return edges

    def _load_kg(self, fpath, name):
        return self._load_array(fpath, name)

    def _load_test_negative(self, fpath, name):
        npy_file = fpath / "{}.npy".format(name)
        if npy_file.exists():
            test_negative = np.load(npy_file)
        else:
            N = 100
            train_ui_set = set(map(tuple, self.train_edgelist))
            test_ui_set = set(map(tuple, self.test_edgelist))
            ui_set = train_ui_set | test_ui_set
            test_negative = np.empty((self.n_users, N), dtype=np.int32)
            for uid in range(self.n_users):
                ui = uid+self.n_entities
                items = []
                while(len(items) < N):
                    i = np.random.randint(self.n_items)
                    if ((ui, i) not in ui_set) and (i not in items):
                        items.append(i)
                test_negative[uid] = items
            np.save(npy_file, test_negative)
        return test_negative

    def ui2kgid(self, edgelist):
        res = np.copy(edgelist)
        res[:, 0] += self.n_entities
        return res

    def negative_sample_kg(self, data):
        def edge2id(h, r, t):
            return r*self.n_entities**2+h*self.n_entities+t

        if self.kg_id is None:
            self.kg_id = edge2id(self.kg[:, 0], self.kg[:, 1], self.kg[:, 2])
        ids = np.random.randint(0, self.n_entities, data.shape[0])
        mask = np.isin(edge2id(data[:, 0], data[:, 1], ids), self.kg_id)
        rest = mask.nonzero()[0]
        while rest.shape[0] > 0:
            tmp = np.random.randint(0, self.n_entities, rest.shape[0])
            mask = np.isin(edge2id(data[rest, 0], data[rest, 1], tmp), self.kg_id)
            ids[rest] = tmp
            rest = rest[mask.nonzero()[0]]
        return ids

    def negative_sample(self, data):
        def edge2id(u, i):
            return u*self.n_items+i
        
        if self.train_id is None:
            self.train_id = edge2id(self.train_edgelist[:, 0], self.train_edgelist[:, 1])
        ids = np.random.randint(0, self.n_items, data.shape[0])
        mask = np.isin(edge2id(data[:, 0], ids), self.train_id)
        rest = mask.nonzero()[0]
        while rest.shape[0] > 0:
            tmp = np.random.randint(0, self.n_items, rest.shape[0])
            mask = np.isin(edge2id(data[rest, 0], tmp), self.train_id)
            ids[rest] = tmp
            rest = rest[mask.nonzero()[0]]
        return ids

    def generate_edgeindex(self):
        edges = np.vstack((self.train_edgelist[:, 0]+self.n_entities, self.train_edgelist[:, 1])).T
        edges = np.vstack((self.kg[:, [0, 2]], edges))
        return edges

    def get_full_kg(self):
        edges = self.generate_edgeindex()
        r = np.hstack((self.kg[:, 1], [self.n_relations]*self.train_edgelist.shape[0]))
        self.fkg = np.vstack([edges[:, 0], r, edges[:, 1]]).T
        self.n_relations += 1
        self.n_triples += self.n_train

if __name__ == '__main__':
    d = Dataset('movie')
    for i in d.kg:
        if i[0] == 780:
            print(i)
    #d.negative_sample_kg(d.kg)
    #print(d.negative_sample(d.train_edgelist))
