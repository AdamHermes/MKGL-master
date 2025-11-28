import argparse
import yaml
import easydict
import torch
import numpy as np
from torch_geometric.utils import degree
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, TaskType

# New Imports
from preprocess_new import InductiveKGCDataset, KGCDataset
from new_retriever import ContextRetriever
from new_score_retriever import ScoreRetriever
from LLM import MKGL

def compute_degree_histogram(edge_index, num_nodes):
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.long)
    return torch.bincount(deg, minlength=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default='config/fb15k237.yaml')
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))

    # Load Data
    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'): config_name += '_' + cfg.dataset.version
    file_path = f'data/preprocessed/{config_name}.pkl'
    
    if 'ind' in config_name:
        dataset = InductiveKGCDataset.load(file_path)
        train_graph = dataset.kgdata.inductive_fact_graph 
    else:
        dataset = KGCDataset.load(file_path)
        train_graph = dataset.kgdata.graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graph = train_graph.to(device)
    deg_hist = compute_degree_histogram(train_graph.edge_index, train_graph.num_nodes).to(device)

    # Load Base LLM
    tokenizer = dataset.tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.tokenizer.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    # PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.loraconfig.r,
        lora_alpha=cfg.loraconfig.lora_alpha,
        target_modules=cfg.loraconfig.target_modules
    )
    model = get_peft_model(base_model, peft_config)

    # Initialize Retrievers
    # Construct KGL token map (KG ID -> Token IDs)
    text_token_list = dataset.vocab_df['text_token_ids'].tolist()
    max_len = cfg.kgl_token_length
    padded = np.zeros((len(text_token_list), max_len), dtype=np.int64)
    for i, t in enumerate(text_token_list):
        trunc = t[:max_len]
        padded[i, :len(trunc)] = trunc
    kgl2token_ids = torch.tensor(padded, device=device)

    ctx_retriever = ContextRetriever(
        cfg.context_retriever, model.get_input_embeddings(), kgl2token_ids, deg_hist
    ).to(device)
    
    score_retriever = ScoreRetriever(
        cfg.score_retriever, ctx_retriever, deg_hist
    ).to(device)

    # Initialize Main MKGL Module
    orig_vocab_size = tokenizer.vocab_size - len(tokenizer.get_added_vocab())
    mkgl_model = MKGL(model, ctx_retriever, score_retriever, train_graph, orig_vocab_size)

    # Trainer
    training_args = TrainingArguments(**cfg.trainer)
    
    # Use TokenClassification collator to handle padding of labels if necessary
    # Or standard collator if inputs are uniform
    collator = DataCollatorForTokenClassification(tokenizer) 

    trainer = Trainer(
        model=mkgl_model,
        args=training_args,
        train_dataset=dataset.train_data,
        eval_dataset=dataset.valid_data,
        data_collator=collator
    )

    trainer.train()

if __name__ == "__main__":
    main()