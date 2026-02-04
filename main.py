import os, sys, logging, argparse, yaml, easydict, json
from datetime import datetime
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

# Import hybrid retriever
try:
    from hybrid_gnn import HybridScoreRetriever, create_hybrid_retriever
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("Warning: hybrid_gnn module not available. Using default retriever.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str,
                        default='')
    parser.add_argument("--seed", "-s", type=str,
                        default=42)
    args = parser.parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if args.version:
            cfg.dataset.version = args.version
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

    kgl2token = torch.tensor(np.stack(dataset.vocab_df.text_token_ids)[:, :cfg.kgl_token_length])     
    model.init_kg_specs(kgl2token, tokenizer.vocab_size, cfg) 
    
    # Check if hybrid retriever should be used
    use_hybrid = cfg.score_retriever.get('use_hybrid', False)
    hybrid_retriever = None
    if use_hybrid and HYBRID_AVAILABLE:
        if rank == 0:
            print("=" * 50)
            print("🚀 Using HybridScoreRetriever (Local PNA + Global GT)")
            print("=" * 50)
        
        # Get the base MKGL model (unwrap PEFT wrapper)
        base_model = model.get_base_model()
        
        # Replace score_retriever with hybrid version
        device = base_model.lm_head.weight.device
        hybrid_retriever = create_hybrid_retriever(
            config=cfg.score_retriever,
            text_embeddings=base_model.lm_head.weight.data,
            kgl2token=kgl2token,
            orig_vocab_size=tokenizer.vocab_size,
        ).to(device)
        
        # Set on the base model so forward() can access it
        base_model.score_retriever = hybrid_retriever
        
        # CRITICAL: Also set on the PEFT wrapper so task can access it
        model.score_retriever = hybrid_retriever
        
        # Mark hybrid retriever parameters as trainable (CRITICAL for PEFT)
        for param in hybrid_retriever.parameters():
            param.requires_grad = True
        
        if rank == 0:
            hybrid_params = sum(p.numel() for p in hybrid_retriever.parameters())
            trainable_params = sum(p.numel() for p in hybrid_retriever.parameters() if p.requires_grad)
            print(f"   Hybrid retriever parameters: {hybrid_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
    elif use_hybrid and not HYBRID_AVAILABLE:
        if rank == 0:
            print("⚠️ Hybrid retriever requested but not available. Using default.")
    
    if rank == 0:
        print(model.print_trainable_parameters())
        print(model)

    
    if 'ind' in args.config:
        task = KGL4IndKGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    else:
        task = KGL4KGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    
    # CRITICAL: Register hybrid retriever as a submodule of task for optimizer visibility
    if use_hybrid and HYBRID_AVAILABLE and hybrid_retriever is not None:
        # Register as a named module so nn.Module.parameters() finds it
        task.register_module('hybrid_retriever', hybrid_retriever)
        if rank == 0:
            print(f"✅ Hybrid retriever registered with task for training")
            
            # Verify it's in task's parameters
            task_param_ids = {id(p) for p in task.parameters()}
            hybrid_param_ids = {id(p) for p in hybrid_retriever.parameters()}
            overlap = len(task_param_ids & hybrid_param_ids)
            print(f"   Hybrid params visible to optimizer: {overlap} / {len(hybrid_param_ids)}")
            
            if overlap < len(hybrid_param_ids):
                print(f"   ⚠️  WARNING: Only {overlap}/{len(hybrid_param_ids)} hybrid params visible!")

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

    # Configure prediction logging
    pred_log_cfg = cfg.get('prediction_logging', {})
    log_predictions = pred_log_cfg.get('enabled', False)
    top_k = pred_log_cfg.get('top_k', 30)
    log_scores = pred_log_cfg.get('log_scores', True)
    
    if log_predictions:
        log_file = pred_log_cfg.get('log_file', 'outputs/predictions.jsonl')
        log_file = log_file.replace('{config_name}', config_name)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        task.enable_prediction_logging(top_k=top_k, log_scores=log_scores)
        if rank == 0:
            print(f"📝 Prediction logging enabled: top-{top_k} predictions will be saved to {log_file}")

    trainer = Trainer(
        model=task,
        args=training_args,
        eval_dataset=dataset.test_data.remove_columns(
            removed_columns),  
        train_dataset=dataset.train_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics
    )
    
    # Initial evaluation
    trainer.evaluate()
    
    # Save predictions after initial eval if logging is enabled
    if log_predictions and rank == 0:
        task.save_predictions(log_file.replace('.jsonl', '_before_train.jsonl'), dataset)
        task.clear_predictions()
    
    # Training
    trainer.train()
    
    # Final evaluation after training
    trainer.evaluate()
    
    # Save final predictions
    if log_predictions and rank == 0:
        task.save_predictions(log_file.replace('.jsonl', '_after_train.jsonl'), dataset)
        print(f"✅ Predictions saved to {log_file.replace('.jsonl', '_after_train.jsonl')}")