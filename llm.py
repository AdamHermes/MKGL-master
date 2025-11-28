import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class MKGL(nn.Module):
    """
    The unified MKGL Model.
    1. Input Processing: Detects <rdf: entity> tokens -> Context Retriever.
    2. Context Processing: Runs LLM (PEFT/LoRA).
    3. Output Processing: Uses LLM hidden state -> Score Retriever -> Entity Logits.
    """
    def __init__(self, llm_model, context_retriever, score_retriever, kg_graph, orig_vocab_size):
        super().__init__()
        self.llm = llm_model
        self.context_retriever = context_retriever
        self.score_retriever = score_retriever
        
        # Registered buffers for graph data (kept on same device as model)
        self.register_buffer("edge_index", kg_graph.edge_index)
        self.register_buffer("kg_nodes", torch.arange(kg_graph.num_nodes))
        
        self.orig_vocab_size = orig_vocab_size
        self.loss_fct = nn.BCEWithLogitsLoss() # Contrastive/Binary loss for ranking

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None, # In MKGL, labels are usually the ID of the true tail entity
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # ==========================================
        # 1. Context Retrieval (Input Injection)
        # ==========================================
        
        # Standard embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Detect KG tokens (those added after original vocab)
        kg_mask = input_ids >= self.orig_vocab_size
        
        if kg_mask.any():
            kg_token_ids = input_ids[kg_mask]
            # Map back to KG IDs (0 to N)
            kg_ids = kg_token_ids - self.orig_vocab_size
            
            # Retrieve dynamic embeddings
            # Note: We pass the full graph edge_index. 
            # PyG PNAConv handles the indexing internally if x is full size, 
            # or we map kg_ids to a subgraph. 
            # Here assumes 'kg_ids' are valid indices into the graph nodes.
            retrieved_embs = self.context_retriever(kg_ids, self.edge_index)
            
            # Inject into the embedding tensor
            inputs_embeds[kg_mask] = retrieved_embs.to(inputs_embeds.dtype)

        # ==========================================
        # 2. LLM Forward Pass
        # ==========================================
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Get the last hidden state of the last token (the query for the tail)
        # (Batch, Hidden)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]

        # ==========================================
        # 3. Score Retrieval (Tail Prediction)
        # ==========================================
        
        if labels is not None:
            # TRAINING: Contrastive Loss
            # We score the True Tail vs a set of Negatives.
            # In the new logic, we score ALL entities (or a sampled subset) efficiently.
            
            # For this implementation, let's score all nodes in the graph (or a subgraph).
            # We assume labels contains the KG ID of the target tail.
            
            # Run Score Retriever
            # Returns logits for all candidates: (Batch, Num_Candidates) or (Num_Nodes) if batch=1
            all_scores = self.score_retriever(
                last_hidden_state, 
                self.kg_nodes, 
                self.edge_index
            )
            
            # Construct Target Tensor
            # Create a one-hot or multi-hot vector for BCE loss
            target = torch.zeros_like(all_scores)
            
            # We need to map labels (KG IDs) to indices in all_scores
            # If all_scores covers all nodes 0..N, labels are direct indices.
            if labels.dim() > 1: labels = labels.squeeze()
            
            # Only set 1s for valid labels that are within range (ignore padding -100)
            valid_labels = labels[labels >= 0]
            target[valid_labels] = 1.0
            
            loss = self.loss_fct(all_scores, target)
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=all_scores.unsqueeze(0), # Dummy logits shape
                hidden_states=outputs.hidden_states
            )
        else:
            # INFERENCE
            all_scores = self.score_retriever(
                last_hidden_state, 
                self.kg_nodes, 
                self.edge_index
            )
            return CausalLMOutputWithPast(
                logits=all_scores.unsqueeze(0)
            )

    # Required for HuggingFace Trainer to save the model correctly
    def save_pretrained(self, save_directory):
        self.llm.save_pretrained(save_directory)
        # We should also save the retriever weights manually or ensure they are part of the state_dict
        torch.save(self.context_retriever.state_dict(), f"{save_directory}/context_retriever.pt")
        torch.save(self.score_retriever.state_dict(), f"{save_directory}/score_retriever.pt")