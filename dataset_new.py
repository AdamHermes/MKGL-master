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
        
        # Counts (Initialize to 0)
        self._num_transductive_nodes = 0
        self._num_inductive_nodes = 0
        self._num_relations = 0

    @property
    def num_relation(self):
        return self._num_relations
        
    @property
    def num_entity(self):
        return self._num_transductive_nodes

    def _create_pyg_graph(self, triplets, num_nodes, num_relations):
        if len(triplets) == 0:
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_type=torch.empty(0, dtype=torch.long),
                        num_nodes=num_nodes)
            
        tensor_triplets = torch.tensor(triplets, dtype=torch.long)
        edge_index = torch.stack([tensor_triplets[:, 0], tensor_triplets[:, 1]], dim=0)
        edge_attr = tensor_triplets[:, 2]
        x = torch.arange(num_nodes, dtype=torch.long)
        return Data(x =x,edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
        # Bước 1: Quét Vocab & Sort Alphabet
        trans_entities = set()
        ind_entities = set()
        relations = set()

        # Quét file Transductive
        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h, r, t = tokens[:3]
                    trans_entities.add(h)
                    trans_entities.add(t)
                    relations.add(r)
        
        # Quét file Inductive
        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h, r, t = tokens[:3]
                    ind_entities.add(h)
                    ind_entities.add(t)
                    # Quan trọng: Inductive Relation phải nằm trong tập Relation gốc
                    # Ở đây ta cứ add vào set để đảm bảo không crash, nhưng assert logic sau này
                    relations.add(r) 

        # Sort Alphabetical để khớp với logic của LLM
        self.transductive_vocab = sorted(list(trans_entities))
        self.inductive_vocab = sorted(list(ind_entities))
        self.relation_vocab = sorted(list(relations))

        # Tạo Inverse Mapping (Token -> ID)
        self.inv_transductive_vocab = {v: k for k, v in enumerate(self.transductive_vocab)}
        self.inv_inductive_vocab = {v: k for k, v in enumerate(self.inductive_vocab)}
        self.inv_relation_vocab = {v: k for k, v in enumerate(self.relation_vocab)}

        self._num_transductive_nodes = len(self.transductive_vocab)
        self._num_inductive_nodes = len(self.inductive_vocab)
        self._num_relations = len(self.relation_vocab)

        # Bước 2: Load triplets bằng ID đã sort
        triplets = []
        num_samples = []

        # 2.1 Load Transductive
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
                    
                    h = self.inv_transductive_vocab[h_token]
                    t = self.inv_transductive_vocab[t_token]
                    r = self.inv_relation_vocab[r_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 2.2 Load Inductive
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
                    
                    h = self.inv_inductive_vocab[h_token]
                    t = self.inv_inductive_vocab[t_token]

                    r = self.inv_relation_vocab[r_token]
                    
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        # 3. Create Graphs
        # Indices logic: [TransTrain, TransValid, TransTest, IndTrain, IndValid, IndTest]
        idx_trans_train = num_samples[0]
        idx_trans_all = sum(num_samples[:3])
        idx_ind_train_start = sum(num_samples[:3])
        idx_ind_train_end = sum(num_samples[:4]) # Inductive Train end
        
        self.fact_graph = self._create_pyg_graph(
            triplets[:idx_trans_train], self._num_transductive_nodes, self._num_relations
        )
        self.graph = self._create_pyg_graph(
            triplets[:idx_trans_all], self._num_transductive_nodes, self._num_relations
        )
        self.inductive_fact_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:idx_ind_train_end], self._num_inductive_nodes, self._num_relations
        )
        self.inductive_graph = self._create_pyg_graph(
            triplets[idx_ind_train_start:], self._num_inductive_nodes, self._num_relations
        )

        # Slice triplets for getitem/dataloader
        # Dataset logic: self.triplets contains:
        # 1. Transductive (Train + Valid)
        # 2. Inductive (Valid + Test)  <-- Note: Inductive Train is usually ONLY in the graph, not evaluated on directly in standard loops unless specified
        
        # num_samples: [TrTr, TrVal, TrTest, InTr, InVal, InTest]
        # slice 1: TrTr + TrVal
        slice_1 = triplets[:sum(num_samples[:2])] 
        
        # slice 2: Inductive Valid + Test (Skip Inductive Train for evaluation triplets)
        # Inductive Train is at index 3. Valid is 4. Test is 5.
        start_idx = sum(num_samples[:4]) # Skip TrTr, TrVal, TrTest, InTr
        slice_2 = triplets[start_idx:] 
        
        self.triplets = torch.tensor(slice_1 + slice_2, dtype=torch.long)
        
        # Update num_samples list to match the sliced triplets structure
        # [TrTrain, TrValid, InValid + InTest]
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

class StandardKGCDataset(InductiveKnowledgeGraphDataset):
    """Base class for standard (transductive) datasets."""
    
    def load_standard_tsvs(self, files, verbose=0):
        # 1. Collect & Sort
        entities = set()
        relations = set()
        for txt_file in files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                for tokens in reader:
                    if len(tokens) < 3: continue
                    h, r, t = tokens[:3]
                    entities.add(h); entities.add(t); relations.add(r)

        self.transductive_vocab = sorted(list(entities))
        self.relation_vocab = sorted(list(relations))
        self.inductive_vocab = [] 
        
        self.inv_transductive_vocab = {v: k for k, v in enumerate(self.transductive_vocab)}
        self.inv_relation_vocab = {v: k for k, v in enumerate(self.relation_vocab)}
        
        self._num_transductive_nodes = len(self.transductive_vocab)
        self._num_relations = len(self.relation_vocab)

        # 2. Load Triplets
        triplets = []
        num_samples = []
        for txt_file in files:
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
                    h = self.inv_transductive_vocab[h_token]
                    t = self.inv_transductive_vocab[t_token]
                    r = self.inv_relation_vocab[r_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        self.fact_graph = self._create_pyg_graph(
            triplets[:num_samples[0]], self._num_transductive_nodes, self._num_relations
        )
        self.graph = self._create_pyg_graph(
            triplets, self._num_transductive_nodes, self._num_relations
        )
        self.inductive_fact_graph = None
        self.inductive_graph = None

        self.triplets = torch.tensor(triplets, dtype=torch.long)
        self.num_samples = num_samples

# --- Inductive Classes ---

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
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        trans_files, ind_files = [], []
        for url in self.transductive_urls:
            url = url % version
            save_file = f"fb15k237_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            trans_files.append(txt_file)
            
        for url in self.inductive_urls:
            url = url % version
            save_file = f"fb15k237_{version}_ind_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            ind_files.append(txt_file)

        self.load_inductive_tsvs(trans_files, ind_files, verbose=verbose)


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
        if not os.path.exists(path): os.makedirs(path)
        
        trans_files, ind_files = [], []
        for url in self.transductive_urls:
            url = url % version
            save_file = f"wn18rr_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                download_url(url, path, filename=save_file)
            trans_files.append(txt_file)
            
        for url in self.inductive_urls:
            url = url % version
            save_file = f"wn18rr_{version}_ind_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                download_url(url, path, filename=save_file)
            ind_files.append(txt_file)

        self.load_inductive_tsvs(trans_files, ind_files, verbose=verbose)


# --- Standard Classes (The missing ones) ---

class FB15k237(StandardKGCDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        files = []
        for url in self.urls:
            url = url % version
            save_file = f"fb15k237_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            files.append(txt_file)

        self.load_standard_tsvs(files, verbose=verbose)


class WN18RR(StandardKGCDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    def __init__(self, path="data/datasets", version="v1", verbose=1):
        super().__init__()
        if not os.path.exists(path): os.makedirs(path)
        
        files = []
        for url in self.urls:
            url = url % version
            save_file = f"wn18rr_{version}_{os.path.basename(url)}"
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                print(f"Downloading {url}...")
                download_url(url, path, filename=save_file)
            files.append(txt_file)

        self.load_standard_tsvs(files, verbose=verbose)