import os, sys, logging, argparse, yaml, easydict
import os
os.environ["MPLBACKEND"] = "Agg"   # MUST be before importing matplotlib or torchdrug

import matplotlib
matplotlib.use("Agg")              # double-force safe backend
import numpy as np
import torch

from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers.trainer import Trainer
from peft import (
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator

import yaml


from llm import *
from collector import *
from preprocess_new import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str,
                        default='')
    parser.add_argument("--seed", "-s", type=str,
                        default=42)
    # Ablation flags for semantic attention
    parser.add_argument("--disable_semantic", action="store_true",
                        help="Disable semantic attention module (for ablation testing)")
    parser.add_argument("--semantic_k", type=int, default=None,
                        help="Override number of semantic neighbors (k)")
    parser.add_argument("--semantic_threshold", type=float, default=None,
                        help="Override semantic similarity threshold")
    args = parser.parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if args.version:
            cfg.dataset.version = args.version
    
    # Apply ablation overrides
    if args.disable_semantic:
        if 'semantic_attention' not in cfg:
            cfg.semantic_attention = easydict.EasyDict({})
        cfg.semantic_attention.use = False
        if rank == 0:
            print("[Main] Semantic attention DISABLED via --disable_semantic flag")
    
    if args.semantic_k is not None:
        if 'semantic_attention' not in cfg:
            cfg.semantic_attention = easydict.EasyDict({})
        cfg.semantic_attention.k = args.semantic_k
        if rank == 0:
            print(f"[Main] Semantic k overridden to: {args.semantic_k}")
    
    if args.semantic_threshold is not None:
        if 'semantic_attention' not in cfg:
            cfg.semantic_attention = easydict.EasyDict({})
        cfg.semantic_attention.threshold = args.semantic_threshold
        if rank == 0:
            print(f"[Main] Semantic threshold overridden to: {args.semantic_threshold}")
    
    torch.manual_seed(args.seed + rank)

    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'):
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name
    cfg.trainer.output_dir += config_name
    
    
    if rank == 0:
        print("Config file: %s" % args.config)
        print(yaml.dump(cfg, sort_keys=False))
    

    saved_dir = 'data/preprocessed/'
    file_path = saved_dir+args.config_name+'.pkl'
    if 'ind' in args.config_name:
        dataset = InductiveKGCDataset.load(file_path)
    else:
        dataset = KGCDataset.load(file_path)
    tokenizer = dataset.tokenizer
    num_rel = int(dataset.kgdata.num_relation)
    cfg.context_retriever.kg_encoder.num_relation = num_rel
    cfg.context_retriever.kg_encoder.num_relations = num_rel 
    cfg.score_retriever.kg_encoder.num_relation = num_rel
    cfg.score_retriever.kg_encoder.num_relations = num_rel
    
    #torch.nn.Module = torch.nn._Module
    config = MKGLConfig.from_pretrained(**cfg.mkglconfig)
    model = MKGL.from_pretrained(
        **cfg.mkgl, device_map={"": Accelerator().process_index}, config=config)

    lora_config = LoraConfig(**cfg.loraconfig)
    model = get_peft_model(model, lora_config)

    # Determine semantic neighbors file path
    semantic_neighbors_path = None
    semantic_cfg = cfg.get('semantic_attention', {})
    if semantic_cfg.get('use', False):
        # Try config-specified path first, then auto-detect
        if semantic_cfg.get('neighbors_file'):
            semantic_neighbors_path = semantic_cfg.neighbors_file
        else:
            # Auto-detect based on config name
            semantic_neighbors_path = f'data/semantic_neighbors_{args.config_name}.pt'
        
        if rank == 0:
            if os.path.exists(semantic_neighbors_path):
                print(f"[Main] Found semantic neighbors file: {semantic_neighbors_path}")
            else:
                print(f"[Main] Warning: Semantic neighbors file not found: {semantic_neighbors_path}")
                print(f"[Main] Run: python scripts/build_semantic_index.py --config {args.config}")

    kgl2token = torch.tensor(np.stack(dataset.vocab_df.text_token_ids)[:, :cfg.kgl_token_length])     
    model.init_kg_specs(kgl2token, tokenizer.vocab_size, cfg, semantic_neighbors_path=semantic_neighbors_path) 
    
    if rank == 0:
        print(model.print_trainable_parameters())
        print(model)

    
    if 'ind' in args.config:
        task = KGL4IndKGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    else:
        task = KGL4KGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    

    data_loader = MKGLDataCollector(dataset)
    
    training_args = TrainingArguments(**cfg.trainer)
    if rank == 0:
        print(training_args)


    def compute_metrics(predictions):
        ranking = predictions[0].astype(float)
        metric = ("mr", "mrr", "hits@1", "hits@3", "hits@10")
        results = {}
        for _metric in metric:
            if _metric == "mr":
                score = ranking.mean()
            elif _metric == "mrr":
                score = (1 / ranking).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            results[_metric] = score
        if rank == 0:
            print(results)
        return results

    removed_columns = ['h_raw', 't_raw', 'r_raw', 'h_fine', 't_fine', 'r_fine', 'inv_r_fine']

    trainer = Trainer(
        model=task,
        args=training_args,
        eval_dataset=dataset.test_data.remove_columns(
            removed_columns),  
        train_dataset=dataset.train_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics
    )
    trainer.evaluate()
    trainer.train()