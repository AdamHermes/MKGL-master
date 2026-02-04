#!/usr/bin/env python3
"""
Diagnostic script to check if hybrid retriever is being trained properly.
Run this after setting up the model in main.py to verify training setup.
"""

import torch
import sys
sys.path.insert(0, '.')

def check_model_parameters(model, task):
    """Check which parameters are trainable."""
    print("\n" + "=" * 80)
    print("PARAMETER ANALYSIS")
    print("=" * 80)
    
    # Check all task parameters
    task_params = list(task.parameters())
    task_trainable = [p for p in task_params if p.requires_grad]
    
    print(f"\nTask total parameters: {sum(p.numel() for p in task_params):,}")
    print(f"Task trainable parameters: {sum(p.numel() for p in task_trainable):,}")
    
    # Check base model
    base_model = model.get_base_model()
    
    # Check score_retriever specifically
    if hasattr(base_model, 'score_retriever') and base_model.score_retriever is not None:
        sr = base_model.score_retriever
        sr_params = list(sr.parameters())
        sr_trainable = [p for p in sr_params if p.requires_grad]
        
        print(f"\nScore Retriever type: {type(sr).__name__}")
        print(f"Score Retriever total parameters: {sum(p.numel() for p in sr_params):,}")
        print(f"Score Retriever trainable parameters: {sum(p.numel() for p in sr_trainable):,}")
        
        # Check if it's in task's parameter iterator
        sr_param_ids = {id(p) for p in sr_params}
        task_param_ids = {id(p) for p in task_params}
        overlap = sr_param_ids & task_param_ids
        
        print(f"\nScore Retriever params in task.parameters(): {len(overlap)} / {len(sr_param_ids)}")
        if len(overlap) < len(sr_param_ids):
            print("⚠️  WARNING: Score retriever parameters NOT fully visible to optimizer!")
            print(f"   Missing {len(sr_param_ids) - len(overlap)} parameter tensors from task.parameters()")
    else:
        print("\n❌ ERROR: score_retriever is None or doesn't exist!")
    
    # Check context_retriever
    if hasattr(base_model, 'context_retriever'):
        cr = base_model.context_retriever
        cr_params = list(cr.parameters())
        cr_trainable = [p for p in cr_params if p.requires_grad]
        print(f"\nContext Retriever trainable: {sum(p.numel() for p in cr_trainable):,}")
    
    # Check LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    print(f"\nLoRA trainable parameters: {sum(p.numel() for p in lora_params):,}")
    
    print("\n" + "=" * 80)


def check_forward_pass(task, batch):
    """Test a forward pass and check gradients."""
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)
    
    # Enable training mode
    task.train()
    
    # Forward pass
    try:
        loss, metrics = task(batch)
        print(f"✅ Forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        
        # Check if loss requires grad
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        # Backward pass
        loss.backward()
        print(f"✅ Backward pass successful")
        
        # Check gradients on score_retriever
        base_model = task.llmodel.get_base_model()
        if hasattr(base_model, 'score_retriever') and base_model.score_retriever is not None:
            sr = base_model.score_retriever
            has_grads = []
            no_grads = []
            
            for name, param in sr.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grads.append((name, param.grad.abs().mean().item()))
                else:
                    no_grads.append(name)
            
            print(f"\n   Score Retriever gradient check:")
            print(f"   - Parameters with gradients: {len(has_grads)}")
            print(f"   - Parameters without gradients: {len(no_grads)}")
            
            if has_grads:
                print(f"\n   Sample gradients:")
                for name, grad_mean in has_grads[:5]:
                    print(f"      {name}: {grad_mean:.6f}")
            
            if no_grads:
                print(f"\n   ⚠️  Parameters without gradients:")
                for name in no_grads[:10]:
                    print(f"      {name}")
        
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


def check_hybrid_components(model):
    """Check if hybrid GNN components are properly initialized."""
    print("\n" + "=" * 80)
    print("HYBRID GNN COMPONENT CHECK")
    print("=" * 80)
    
    base_model = model.get_base_model()
    if hasattr(base_model, 'score_retriever') and base_model.score_retriever is not None:
        sr = base_model.score_retriever
        
        # Check for hybrid-specific attributes
        if hasattr(sr, 'hybrid_retriever'):
            print(f"✅ Hybrid retriever found")
            hr = sr.hybrid_retriever
            
            # Check components
            if hasattr(hr, 'sampler'):
                print(f"   - Sampler: {type(hr.sampler).__name__}")
            if hasattr(hr, 'pearl_gin'):
                print(f"   - PEARL GIN: {hr.pearl_gin.num_layers} layers")
            if hasattr(hr, 'hybrid_blocks'):
                print(f"   - Hybrid Blocks: {len(hr.hybrid_blocks)} blocks")
            if hasattr(hr, 'scorer'):
                print(f"   - Scorer: {type(hr.scorer).__name__}")
        else:
            print(f"⚠️  score_retriever is {type(sr).__name__}, not HybridScoreRetriever")
    else:
        print(f"❌ No score_retriever found")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print("""
This is a diagnostic helper script. Import it in main.py after model setup:

    from scripts.diagnose_training import check_model_parameters, check_forward_pass, check_hybrid_components
    
    # After creating task
    check_hybrid_components(model)
    check_model_parameters(model, task)
    
    # Before trainer.train(), test one batch
    # sample_batch = next(iter(trainer.get_train_dataloader()))
    # check_forward_pass(task, sample_batch)
""")
