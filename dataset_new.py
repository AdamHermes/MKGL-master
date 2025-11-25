import os
import csv
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, download_url
from tqdm import tqdm

class InductiveKnowledgeGraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.fact_graph = None
        self.graph = None
        self.inductive_fact_graph = None
        self.inductive_graph = None
        self.triplets = None
        self.num_samples = []
        
        # Vocabularies
        self.transductive_vocab = []
        self.inductive_vocab = []
        self.relation_vocab = []
        self.inv_transductive_vocab = {}
        self.inv_inductive_vocab = {}
        self.inv_relation_vocab = {}

    def _create_pyg_graph(self, triplets, num_nodes, num_relations):
        """
        Helper to convert a list of (h, t, r) triplets into a PyG Data object.
        """
        if len(triplets) == 0:
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_type=torch.empty(0, dtype=torch.long),
                        num_nodes=num_nodes)
            
        tensor_triplets = torch.tensor(triplets, dtype=torch.long)
        # PyG expects edge_index of shape [2, num_edges] -> [source, target]
        # In KG triplets (h, t, r), h is source, t is target.
        edge_index = torch.stack([tensor_triplets[:, 0], tensor_triplets[:, 1]], dim=0)
        edge_type = tensor_triplets[:, 2]
        
        return Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)

    def _finalize_vocab(self, inv_vocab):
        """Replaces TorchDrug's _standarize_vocab"""
        # Sort by ID to ensure the list matches the index
        sorted_items = sorted(inv_vocab.items(), key=lambda x: x[1])
        vocab_list = [k for k, v in sorted_items]
        return vocab_list, inv_vocab

    def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
        assert len(transductive_files) == len(inductive_files) == 3
        
        inv_transductive_vocab = {}
        inv_inductive_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        # 1. Load Transductive Files
        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(list(reader), desc=f"Loading {os.path.basename(txt_file)}")
                else:
                    reader = list(reader)

                num_sample = 0
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h_token, r_token, t_token = tokens[:3]
                    
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 2. Load Inductive Files
        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(list(reader), desc=f"Loading {os.path.basename(txt_file)}")
                else:
                    reader = list(reader)

                num_sample = 0
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h_token, r_token, t_token = tokens[:3]
                    
                    if h_token not in inv_inductive_vocab:
                        inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
                    h = inv_inductive_vocab[h_token]
                    
                    # Relation vocab is shared and fixed from transductive phase
                    assert r_token in inv_relation_vocab, f"Unknown relation {r_token} in inductive set"
                    r = inv_relation_vocab[r_token]
                    
                    if t_token not in inv_inductive_vocab:
                        inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
                    t = inv_inductive_vocab[t_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 3. Finalize Vocabs
        self.transductive_vocab, self.inv_transductive_vocab = self._finalize_vocab(inv_transductive_vocab)
        self.inductive_vocab, self.inv_inductive_vocab = self._finalize_vocab(inv_inductive_vocab)
        self.relation_vocab, self.inv_relation_vocab = self._finalize_vocab(inv_relation_vocab)
        
        self._num_transductive_nodes = len(self.transductive_vocab)
        self._num_inductive_nodes = len(self.inductive_vocab)
        self._num_relations = len(self.relation_vocab)
        
        num_trans_nodes = len(self.transductive_vocab)
        num_ind_nodes = len(self.inductive_vocab)
        num_relations = len(self.relation_vocab)
        


        # 4. Create PyG Graph Objects
        # Slice indices based on how the files were read:
        # 0: Trans-Train, 1: Trans-Valid, 2: Trans-Test
        # 3: Ind-Train,   4: Ind-Valid,   5: Ind-Test
        
        idx_trans_train = num_samples[0]
        idx_trans_all = sum(num_samples[:3])
        idx_ind_train_start = sum(num_samples[:3])
        idx_ind_train_end = sum(num_samples[:4])
        
        # Fact Graph (Transductive Training)
        self.fact_graph = self._create_pyg_graph(
            triplets[:idx_trans_train], num_trans_nodes, num_relations
        )
        
        # Full Transductive Graph
        self.graph = self._create_pyg_graph(
            triplets[:idx_trans_all], num_trans_nodes, num_relations
        )
        
        # Inductive Fact Graph (Inductive Training)
        self.inductive_fact_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:idx_ind_train_end], num_ind_nodes, num_relations
        )
        
        # Full Inductive Graph
        self.inductive_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:], num_ind_nodes, num_relations
        )

        # Store all triplets as a tensor for __getitem__
        # We skip the transductive valid/test triplets in the main list, matching original logic?
        # Original: triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:4]):]
        # This includes: Trans-Train, Trans-Valid (index 0, 1) AND Ind-Valid, Ind-Test (index 4, 5)
        # It seems to skip Trans-Test (index 2) and Ind-Train (index 3) from the "triplets" list used for iteration?
        # Replicating original logic exactly:
        
        slice_1 = triplets[:sum(num_samples[:2])] # Trans Train + Valid
        slice_2 = triplets[sum(num_samples[:4]):] # Ind Valid + Test
        
        self.triplets = torch.tensor(slice_1 + slice_2, dtype=torch.long)
        
        # Num samples for splitting logic later
        # Trans-Train, Trans-Valid, Ind-Valid+Test
        self.num_samples = num_samples[:2] + [sum(num_samples[4:])]

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    @property
    def num_relation(self):
        return self._num_relations
        
    @property
    def num_entity(self):
        # Depending on split, this might need to return transductive or inductive count
        # Usually for model init, we care about the transductive (base) count
        return self._num_transductive_nodes

class FB15k237Inductive(InductiveKnowledgeGraphDataset):
    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        # Ensure parent init is called to set up structures
        super().__init__()
        
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url} to {txt_file}...")
                download_url(url, path, filename=save_file)
            transductive_files.append(txt_file)
            
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url} to {txt_file}...")
                download_url(url, path, filename=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


class WN18RRInductive(InductiveKnowledgeGraphDataset):
    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url} to {txt_file}...")
                download_url(url, path, filename=save_file)
            transductive_files.append(txt_file)
            
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url} to {txt_file}...")
                download_url(url, path, filename=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)