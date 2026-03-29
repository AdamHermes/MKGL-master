import argparse
import json
import os
import os.path as osp
import pickle
from pathlib import Path
import yaml
import easydict
import numpy as np
import pandas as pd
import swifter
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from dataset_new import FB15k237Inductive, WN18RRInductive, FB15k237, WN18RR
# Import the PyG datasets we migrated


def _default_multimodal_config():
    return easydict.EasyDict({
        "use_images": True,
        "image_dir": "image-graph_urls",
        "image_index_file": "FB15K_ImageIndex.txt",
        "image_feature_file": "FB15K_ImageData.h5",
        "image_feature_dim": 4096,
        "image_token_prefix": "<img: ",
    })


def get_multimodal_config(cfg=None):
    multimodal_cfg = _default_multimodal_config()
    if cfg is None:
        return multimodal_cfg

    user_cfg = cfg.get("multimodal", {})
    for key, value in user_cfg.items():
        multimodal_cfg[key] = value
    return multimodal_cfg


def load_fb15k_image_index(multimodal_cfg):
    image_dir = Path(multimodal_cfg.image_dir)
    index_path = image_dir / multimodal_cfg.image_index_file
    if not index_path.exists():
        return pd.DataFrame(columns=["raw_name", "image_id"])

    image_index_df = pd.read_csv(
        index_path,
        sep="\t",
        header=None,
        names=["raw_name", "image_id"],
        dtype=str,
    )
    image_index_df = image_index_df.dropna(subset=["raw_name", "image_id"])
    return image_index_df


def load_entity_image_features(image_raw_names, multimodal_cfg):
    if not image_raw_names:
        feature_dim = int(multimodal_cfg.get("image_feature_dim", 4096))
        return (
            torch.zeros((0, feature_dim), dtype=torch.float16),
            torch.zeros(0, dtype=torch.bool),
        )

    import h5py

    image_index_df = load_fb15k_image_index(multimodal_cfg)
    image_id_by_raw_name = pd.Series(
        image_index_df["image_id"].values,
        index=image_index_df["raw_name"].values,
    )

    feature_dim = int(multimodal_cfg.get("image_feature_dim", 4096))
    features = np.zeros((len(image_raw_names), feature_dim), dtype=np.float32)
    has_image = np.zeros(len(image_raw_names), dtype=bool)

    feature_path = Path(multimodal_cfg.image_dir) / multimodal_cfg.image_feature_file
    if not feature_path.exists():
        return (
            torch.tensor(features, dtype=torch.float16),
            torch.tensor(has_image, dtype=torch.bool),
        )

    with h5py.File(feature_path, "r") as h5_file:
        for row_idx, raw_name in enumerate(image_raw_names):
            image_id = image_id_by_raw_name.get(raw_name)
            if image_id is None or image_id not in h5_file:
                continue

            image_vector = np.asarray(h5_file[image_id], dtype=np.float32).reshape(-1)
            if image_vector.shape[0] != feature_dim:
                raise ValueError(
                    f"Unexpected image feature dim for {raw_name}: "
                    f"{image_vector.shape[0]} != {feature_dim}"
                )

            features[row_idx] = image_vector
            has_image[row_idx] = True

    return (
        torch.tensor(features, dtype=torch.float16),
        torch.tensor(has_image, dtype=torch.bool),
    )


def build_kg_token_tables(dataset, kgl_token_length):
    orig_vocab_size = dataset.tokenizer.vocab_size
    text_vocab_df = dataset.vocab_df.sort_index()
    image_vocab_df = getattr(dataset, "image_vocab_df", pd.DataFrame()).sort_index()

    token_ids = []
    if len(text_vocab_df):
        token_ids.extend(text_vocab_df.index.tolist())
    if len(image_vocab_df):
        token_ids.extend(image_vocab_df.index.tolist())

    if not token_ids:
        raise ValueError("No KG tokens found in dataset vocabulary.")

    max_token_id = max(token_ids)
    num_added_tokens = max_token_id - orig_vocab_size + 1

    text_kgl2token = np.zeros((num_added_tokens, kgl_token_length), dtype=np.int64)
    kg_token_type = np.zeros(num_added_tokens, dtype=np.int64)
    image_kgl2index = np.full(num_added_tokens, -1, dtype=np.int64)

    for token_id, token_ids_per_name in text_vocab_df["text_token_ids"].items():
        offset = int(token_id) - orig_vocab_size
        truncated = np.asarray(token_ids_per_name[:kgl_token_length], dtype=np.int64)
        text_kgl2token[offset, : len(truncated)] = truncated
        kg_token_type[offset] = 1

    if len(image_vocab_df):
        reset_image_vocab_df = image_vocab_df.reset_index()
        for image_row_idx, row in reset_image_vocab_df.iterrows():
            offset = int(row["token_index"]) - orig_vocab_size
            kg_token_type[offset] = 2
            image_kgl2index[offset] = image_row_idx

        image_raw_names = reset_image_vocab_df["raw_name"].tolist()
    else:
        image_raw_names = []

    return {
        "text_kgl2token": torch.tensor(text_kgl2token, dtype=torch.long),
        "kg_token_type": torch.tensor(kg_token_type, dtype=torch.long),
        "image_kgl2index": torch.tensor(image_kgl2index, dtype=torch.long),
        "image_raw_names": image_raw_names,
        "orig_vocab_size": orig_vocab_size,
    }


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: str = None,
        label: str = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class InductiveKGCDataset(object):
    """
    Wrapper class that takes a raw PyG dataset (kgdata), tokenizes it, 
    and adds the instruction tuning text prompts.
    """
    def __init__(self, args, kgdata, tokenizer, cfg=None):
        self.args = args
        self.kgdata = kgdata
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.multimodal_cfg = get_multimodal_config(cfg)
        self.prompter = Prompter('alpaca_short', verbose=False)
        self.inv_prefix = '/inv'
        self.inv_fine_prefix = 'inverse of '

        self.read_vocab()
        self.read_data()
        self.add_input_text()
        self.post_process()

        self.saved_dir = 'data/preprocessed/'
        self.save()

    def read_vocab(self):
        kgdata = self.kgdata
        enable_images = bool(self.multimodal_cfg.use_images)

        # Determine name prefix based on config name
        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        elif 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'
        else:
            # Fallback or error
            name_prefix = './data/names/fb15k237/'

        # Read fine-grained descriptions from text files
        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values,
                             index=ent_name['raw_name'].values)
        
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values,
                             index=rel_name['raw_name'].values)

        # Build DataFrames using vocab lists from the PyG dataset
        # Note: kgdata.transductive_vocab is a list of raw strings
        trans_ent_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.transductive_vocab)), 
            'raw_name': kgdata.transductive_vocab, 
            'transductive': 1
        })
        
        ind_ent_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.inductive_vocab)), 
            'raw_name': kgdata.inductive_vocab, 
            'transductive': 0
        })
        
        ent_vocab_df = pd.concat([trans_ent_vocab_df, ind_ent_vocab_df], ignore_index=True)
        
        # Map raw names to fine-grained descriptions
        # Using .get to avoid crashes if a name is missing in entity.txt
        ent_vocab_df['fine_name'] = ent_vocab_df['raw_name'].map(lambda x: ent2text.get(x, x))

        rel_vocab_df = pd.DataFrame({
            'kg_id': range(len(kgdata.relation_vocab)), 
            'raw_name': kgdata.relation_vocab, 
            'transductive': 0
        })
        rel_vocab_df['fine_name'] = rel_vocab_df['raw_name'].map(lambda x: rel2text.get(x, x))

        # Create Inverse Relations
        inv_rel_vocab_df = rel_vocab_df.iloc[:]
        inv_rel_vocab_df = inv_rel_vocab_df.copy() # Avoid SettingWithCopy warning
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat([rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        # Handle overlapped names (same description for different IDs)
        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        # ent_vocab_df = ent_vocab_df.droplevel('fine_name').sort_index() # Not needed with group_keys=False usually, but keeping safe
        
        rel_vocab_df = rel_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        # rel_vocab_df = rel_vocab_df.droplevel('fine_name').sort_index()

        self.entity_vocab_df = ent_vocab_df.copy()

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        # Tokenize
        def tokenize_vocab(df):
            new_tokens = df['token_name'].values.tolist()
            self.tokenizer.add_tokens(new_tokens)
            
            # Get added tokens map
            vocab_map = self.tokenizer.get_added_vocab()
            # FIX: Use get_vocab() instead of .vocab attribute
            base_vocab = self.tokenizer.get_vocab()
            
            # Look up tokens in added vocab first, then base vocab, default to 0
            df['token_index'] = [
                vocab_map.get(tn, base_vocab.get(tn, 0)) 
                for tn in df['token_name'].values
            ]

            # Deduplicate raw_names before building the lookup series.
            # When transductive and inductive vocabs share a raw_name string,
            # the Series index would have duplicates, causing lookups to return
            # a Series instead of a scalar and breaking all downstream assignments.
            # np.unique picks the first occurrence (lowest token_index), matching
            # the original preprocess.py behaviour.
            raw_names_arr = df['raw_name'].values
            token_index_arr = df['token_index'].values
            unique_raw_names, first_indices = np.unique(raw_names_arr, return_index=True)
            rawname2tokenid = pd.Series(
                token_index_arr[first_indices], index=unique_raw_names)

            df.set_index('token_index', inplace=True)
            fine_names = [str(n).strip() for n in df['fine_name'].values]
            
            tokenized = self.tokenizer(
                fine_names, add_special_tokens=False, truncation=True, padding=True
            )
            df['text_token_ids'] = tokenized.input_ids
            return df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize_vocab(vocab_df)

        self.image_vocab_df = pd.DataFrame()
        self.rawname2image_tokenid = pd.Series(dtype=np.int64)
        if enable_images:
            image_index_df = load_fb15k_image_index(self.multimodal_cfg)
            image_id_lookup = pd.Series(
                image_index_df["image_id"].values,
                index=image_index_df["raw_name"].values,
            )

            image_vocab_df = self.entity_vocab_df.drop_duplicates(
                subset=["raw_name"], keep="first"
            ).copy()
            image_vocab_df["entity"] = 1
            image_vocab_df["image_id"] = image_vocab_df["raw_name"].map(
                lambda x: image_id_lookup.get(x, None)
            )
            image_vocab_df["has_image"] = image_vocab_df["image_id"].notna()
            image_vocab_df["token_name"] = (
                self.multimodal_cfg.image_token_prefix + image_vocab_df["fine_name"] + ">"
            )

            image_tokens = image_vocab_df["token_name"].values.tolist()
            self.tokenizer.add_tokens(image_tokens)
            vocab_map = self.tokenizer.get_added_vocab()
            base_vocab = self.tokenizer.get_vocab()
            image_vocab_df["token_index"] = [
                vocab_map.get(token_name, base_vocab.get(token_name, 0))
                for token_name in image_vocab_df["token_name"].values
            ]

            image_vocab_df.set_index("token_index", inplace=True)
            unique_raw_names, first_indices = np.unique(
                image_vocab_df["raw_name"].values, return_index=True
            )
            self.rawname2image_tokenid = pd.Series(
                image_vocab_df.index.values[first_indices], index=unique_raw_names
            )
            self.image_vocab_df = image_vocab_df

    def read_data(self):
        kgdata = self.kgdata
        # PyG Dataset.split() returns a list of Subsets
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)

            # Optimizing for PyG: extract tensors directly using indices
            # subset.dataset is the underlying dataset object
            # subset.indices are the indices for this split
            indices = subset.indices
            triplets = subset.dataset.triplets[indices]
            
            # Convert to numpy for DataFrame creation
            data_np = triplets.cpu().numpy()

            df = pd.DataFrame(data_np, columns=['h_id', 't_id', 'r_id'])
            
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            # Map raw names to token IDs
            # Using .values ensures we pass numpy arrays to map, which is faster
            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            
            # Inverse relation logic
            inv_r_raw = self.inv_prefix + df['r_raw'].values
            df['inv_r_tokenid'] = self.rawname2tokenid[inv_r_raw].values
            if len(self.rawname2image_tokenid):
                df['h_img_tokenid'] = self.rawname2image_tokenid[df['h_raw'].values].values
                df['t_img_tokenid'] = self.rawname2image_tokenid[df['t_raw'].values].values

            # Map token IDs to fine descriptions
            # Using reindex or loc
            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values, 'fine_name'].values

            return df

        train_df = convert_to_df(train_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        valid_df = convert_to_df(valid_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        test_df = convert_to_df(test_set, kgdata.inductive_vocab, kgdata.relation_vocab)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def add_input_text(self):
        print('##########Add input text##########')

        train_df, valid_df, test_df = self.train_df, self.valid_df, self.test_df
        vocab_df = self.vocab_df

        def produce_input_text(row):
            h_info = vocab_df.loc[row['h_tokenid']]
            t_info = vocab_df.loc[row['t_tokenid']]
            r_info = vocab_df.loc[row['r_tokenid']]
            inv_r_info = vocab_df.loc[row['inv_r_tokenid']]
            h_img_info = self.image_vocab_df.loc[row['h_img_tokenid']]
            t_img_info = self.image_vocab_df.loc[row['t_img_tokenid']]

            h_img = h_img_info['token_name']
            h = h_info['token_name']
            t_img = t_img_info['token_name']
            t = t_info['token_name']
            r = r_info['token_name']
            inv_r = inv_r_info['token_name']

            h_img_des = f'Visual representation of {h_info["fine_name"]}'
            h_des = h_info['fine_name']
            t_img_des = f'Visual representation of {t_info["fine_name"]}'
            t_des = t_info['fine_name']
            r_des = r_info['fine_name']
            inv_r_des = inv_r_info['fine_name']

            instruction = (
                "Suppose that you are an excellent linguist studying a three-word language. "
                "Given the following multimodal dictionary:\n\n"
                " Input\tType\tDescription\n"
                f"{h_img}\tHead image\t{h_img_des}\n"
                f"{h}\tHead entity\t{h_des}\n"
                f"{r}\tRelation\t{r_des}\n\n"
                f"Please complete the multimodal phrase: {h_img}{h}{r}?"
            )
            inv_instruction = (
                "Suppose that you are an excellent linguist studying a three-word language. "
                "Given the following multimodal dictionary:\n\n"
                " Input\tType\tDescription\n"
                f"{t_img}\tHead image\t{t_img_des}\n"
                f"{t}\tHead entity\t{t_des}\n"
                f"{inv_r}\tRelation\t{inv_r_des}\n\n"
                f"Please complete the multimodal phrase: {t_img}{t}{inv_r}?"
            )

            row['input_text'] = self.prompter.generate_prompt(instruction, label=f'{h_img}{h}{r}')
            row['inv_input_text'] = self.prompter.generate_prompt(
                inv_instruction, label=f'{t_img}{t}{inv_r}')

            return row

        test_df = test_df.swifter.apply(produce_input_text, axis=1)
        valid_df = valid_df.swifter.apply(produce_input_text, axis=1)
        train_df = train_df.swifter.apply(produce_input_text, axis=1)

        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def _to_hf_dataset(self, df):
        return Dataset.from_pandas(df)

    def post_process(self):
        print('##########Post process: convert to hf datasets##########')
        self.train_data = self._to_hf_dataset(self.train_df)
        self.valid_data = self._to_hf_dataset(self.valid_df)
        self.test_data = self._to_hf_dataset(self.test_df)

    def save(self):
        saved_dir = self.saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file_path = saved_dir + self.args.config_name + '.pkl'
        print('##########Save dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        print('##########Load dataset from %s############' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class KGCDataset(InductiveKGCDataset):
    """
    Standard KGC Dataset Wrapper (Transductive).
    Adaptation for PyG compatibility, though main focus is Inductive.
    """
    def read_vocab(self):
        kgdata = self.kgdata
        enable_images = bool(self.multimodal_cfg.use_images)
        
        # Similar name prefix logic
        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        elif 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'
        else:
            name_prefix = './data/names/fb15k237/'

        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values, index=ent_name['raw_name'].values)
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values, index=rel_name['raw_name'].values)

        # For standard KGC, we use 'transductive_vocab' as the only entity vocab
        # If your dataset.py uses 'transductive_vocab' for standard KGC, use that.
        # Assuming we reuse InductiveKnowledgeGraphDataset for simplicity where transductive_vocab == transductive_vocab
        ent_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.transductive_vocab)), 'raw_name': kgdata.transductive_vocab, 'transductive': 1}, )
        ent_vocab_df['fine_name'] = ent2text[ent_vocab_df['raw_name'].values].values

        rel_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.relation_vocab)), 'raw_name': kgdata.relation_vocab, 'transductive': 0})
        rel_vocab_df['fine_name'] = rel2text[rel_vocab_df['raw_name'].values].values

        inv_rel_vocab_df = rel_vocab_df.iloc[:]
        inv_rel_vocab_df = inv_rel_vocab_df.copy()  # Avoid SettingWithCopyWarning on mutations below
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + \
            inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + \
            inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat(
            [rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)
        rel_vocab_df = rel_vocab_df.groupby('fine_name', group_keys=False).apply(process_overlapped_name)

        self.entity_vocab_df = ent_vocab_df.copy()

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        def tokenize_vocab(df):
            new_tokens = df['token_name'].values.tolist()
            self.tokenizer.add_tokens(new_tokens)
            
            # Get added tokens map
            vocab_map = self.tokenizer.get_added_vocab()
            # FIX: Use get_vocab() instead of .vocab attribute
            base_vocab = self.tokenizer.get_vocab()
            
            # Look up tokens in added vocab first, then base vocab, default to 0
            df['token_index'] = [
                vocab_map.get(tn, base_vocab.get(tn, 0)) 
                for tn in df['token_name'].values
            ]

            # Deduplicate raw_names — same rationale as InductiveKGCDataset.
            raw_names_arr = df['raw_name'].values
            token_index_arr = df['token_index'].values
            unique_raw_names, first_indices = np.unique(raw_names_arr, return_index=True)
            rawname2tokenid = pd.Series(
                token_index_arr[first_indices], index=unique_raw_names)

            df.set_index('token_index', inplace=True)
            fine_names = [str(n).strip() for n in df['fine_name'].values]
            
            tokenized = self.tokenizer(
                fine_names, add_special_tokens=False, truncation=True, padding=True
            )
            df['text_token_ids'] = tokenized.input_ids
            return df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize_vocab(vocab_df)

        self.image_vocab_df = pd.DataFrame()
        self.rawname2image_tokenid = pd.Series(dtype=np.int64)
        if enable_images:
            image_index_df = load_fb15k_image_index(self.multimodal_cfg)
            image_id_lookup = pd.Series(
                image_index_df["image_id"].values,
                index=image_index_df["raw_name"].values,
            )

            image_vocab_df = self.entity_vocab_df.drop_duplicates(
                subset=["raw_name"], keep="first"
            ).copy()
            image_vocab_df["entity"] = 1
            image_vocab_df["image_id"] = image_vocab_df["raw_name"].map(
                lambda x: image_id_lookup.get(x, None)
            )
            image_vocab_df["has_image"] = image_vocab_df["image_id"].notna()
            image_vocab_df["token_name"] = (
                self.multimodal_cfg.image_token_prefix + image_vocab_df["fine_name"] + ">"
            )

            image_tokens = image_vocab_df["token_name"].values.tolist()
            self.tokenizer.add_tokens(image_tokens)
            vocab_map = self.tokenizer.get_added_vocab()
            base_vocab = self.tokenizer.get_vocab()
            image_vocab_df["token_index"] = [
                vocab_map.get(token_name, base_vocab.get(token_name, 0))
                for token_name in image_vocab_df["token_name"].values
            ]

            image_vocab_df.set_index("token_index", inplace=True)
            unique_raw_names, first_indices = np.unique(
                image_vocab_df["raw_name"].values, return_index=True
            )
            self.rawname2image_tokenid = pd.Series(
                image_vocab_df.index.values[first_indices], index=unique_raw_names
            )
            self.image_vocab_df = image_vocab_df
    
    def read_data(self):
        # Override to ensure we look at the right vocab for all splits
        kgdata = self.kgdata
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)
            
            indices = subset.indices
            triplets = subset.dataset.triplets[indices]
            data_np = triplets.cpu().numpy()
            
            df = pd.DataFrame(data_np, columns=['h_id', 't_id', 'r_id'])
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            df['inv_r_tokenid'] = self.rawname2tokenid[self.inv_prefix + df['r_raw'].values].values
            if len(self.rawname2image_tokenid):
                df['h_img_tokenid'] = self.rawname2image_tokenid[df['h_raw'].values].values
                df['t_img_tokenid'] = self.rawname2image_tokenid[df['t_raw'].values].values

            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values, 'fine_name'].values

            return df

        # Use transductive vocab for all splits in standard KGC
        train_df = convert_to_df(train_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        valid_df = convert_to_df(valid_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        test_df = convert_to_df(test_set, kgdata.transductive_vocab, kgdata.relation_vocab)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str, default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str, default='')
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))

        # Match legacy behavior: inductive runs must provide a version
        if 'ind' in args.config:
            if not args.version:
                raise ValueError("Inductive config requires --version (e.g., v1).")
            print(f"Using inductive version from command line: '{args.version}'")
            cfg.dataset.version = args.version
        elif args.version:
            print(f"Overriding dataset version from command line: '{args.version}'")
            cfg.dataset.version = args.version
        elif not hasattr(cfg.dataset, 'version'):
            # Non-inductive configs default to full dataset unless version is set in YAML
            cfg.dataset.version = ''

    # Set Config Name
    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version') and cfg.dataset.version:
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name

    print('***************Read dataset from PyG (Migrated)***************')
    print("Config file: %s" % args.config)
    print("Config name: %s" % args.config_name)
    print("Dataset version: %s" % cfg.dataset.get('version', 'NOT SET'))
    import pprint
    pprint.pprint(cfg)
    
    # Instantiate Dataset (Replaces TorchDrug core.Configurable)
    dataset_class_str = cfg.dataset.get('class', '')
    dataset_version = cfg.dataset.get('version', '')
    is_inductive = 'Inductive' in dataset_class_str or 'ind' in args.config

    if is_inductive and not dataset_version:
        raise ValueError("Inductive datasets need a version. Pass --version or set dataset.version in config.")

    kgdata = None
    # Inductive Check
    if 'FB15k237Inductive' in dataset_class_str:
        kgdata = FB15k237Inductive(version=dataset_version)
    elif 'WN18RRInductive' in dataset_class_str:
        kgdata = WN18RRInductive(version=dataset_version)
    # Standard Check (New!)
    elif 'FB15k237' in dataset_class_str:
        kgdata = FB15k237(version=dataset_version)
    elif 'WN18RR' in dataset_class_str:
        kgdata = WN18RR(version=dataset_version)
    else:
        print(f"Warning: Unknown dataset class {dataset_class_str} in config.")
        if is_inductive:
            raise ValueError("Please ensure dataset.py contains the inductive class requested.")

    print('***************Load tokenizer***************')
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    
    if kgdata:
        if is_inductive:
            dataset = InductiveKGCDataset(args, kgdata, tokenizer, cfg)
        else:
            dataset = KGCDataset(args, kgdata, tokenizer, cfg)
