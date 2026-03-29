import torch
from torch import nn
import torch.nn.functional as F

from gnn2.model import *
from gnn2.layer import PNALayer


class BasePNARetriever(nn.Module):
    """
    Retrieve KG token embeddings from text descriptions and aligned image features.
    """

    def __init__(
        self,
        config,
        text_embeddings,
        text_kgl2token,
        kg_token_type,
        image_kgl2index,
        image_features,
        image_feature_mask,
        orig_vocab_size,
    ):
        super().__init__()
        self.config = config
        self.text_embeddings = text_embeddings
        self.text_kgl2token = text_kgl2token
        self.kg_token_type = kg_token_type
        self.image_kgl2index = image_kgl2index
        self.image_features = image_features
        self.image_feature_mask = image_feature_mask
        self.orig_vocab_size = orig_vocab_size

        self.down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.image_down_scaling = nn.Linear(
            self.config.image_feature_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.missing_image_embedding = nn.Parameter(
            torch.zeros(1, self.config.r, dtype=torch.float)
        )

        if self.config.text_encoder == "pna":
            self.re_scaling = nn.Linear(config.r * 12, self.config.r)

    def aggregate_text(self, token_ids, text_embeddings, method="pna"):
        device = text_embeddings.device

        token_ids = token_ids.to(device)
        token_mask = (token_ids > 0).unsqueeze(-1).to(device)
        token_lengths = token_mask.half().sum(axis=1).to(device).clamp(min=1)
        degree = token_lengths
        token_embs = text_embeddings[token_ids]

        mean = (token_embs * token_mask).sum(axis=1) / token_lengths
        if method == "mean":
            result = mean
        else:
            sq_mean = (token_embs ** 2 * token_mask).sum(axis=1) / token_lengths
            max_emb, _ = (token_embs * token_mask).max(axis=1)
            min_emb, _ = (token_embs * token_mask).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat([mean, max_emb, min_emb, std], dim=-1)

            scale = degree.log()
            scale = scale / scale.mean().clamp(min=1e-6)
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )

            result = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        return result

    def retrieve_text(self, token_ids):
        reduced_embeddings = self.down_scaling(self.text_embeddings)
        result = self.aggregate_text(token_ids, reduced_embeddings, self.config.text_encoder)

        if self.config.text_encoder == "pna":
            result = self.re_scaling(result)

        return self.norm(result)

    def image_token_has_feature(self, image_kgl_ids):
        flat_ids = image_kgl_ids.reshape(-1)
        offsets = flat_ids - self.orig_vocab_size
        image_indices = self.image_kgl2index[offsets]

        valid_mask = image_indices >= 0
        result = torch.zeros_like(valid_mask, dtype=torch.bool)
        if valid_mask.any():
            valid_indices = image_indices[valid_mask]
            result[valid_mask] = self.image_feature_mask[valid_indices]
        return result.reshape(image_kgl_ids.shape)

    def retrieve_image(self, image_kgl_ids):
        flat_ids = image_kgl_ids.reshape(-1)
        offsets = flat_ids - self.orig_vocab_size
        image_indices = self.image_kgl2index[offsets]

        result = self.missing_image_embedding.expand(flat_ids.shape[0], -1).clone()
        valid_mask = image_indices >= 0
        if valid_mask.any():
            valid_indices = image_indices[valid_mask]
            has_feature_mask = self.image_feature_mask[valid_indices]
            if has_feature_mask.any():
                projected = self.image_down_scaling(
                    self.image_features[valid_indices[has_feature_mask]].float()
                )
                valid_positions = valid_mask.nonzero(as_tuple=False).flatten()
                result[valid_positions[has_feature_mask]] = projected

        result = self.norm(result)
        return result.reshape(*image_kgl_ids.shape, -1)

    def norm(self, x):
        return F.normalize(x, p=2, dim=1)

    def forward(self, kgl_ids=None):
        if kgl_ids is None:
            return self.retrieve_text(self.text_kgl2token)

        flat_ids = kgl_ids.reshape(-1)
        offsets = flat_ids - self.orig_vocab_size
        token_types = self.kg_token_type[offsets]
        result = torch.zeros(
            flat_ids.shape[0],
            self.config.r,
            device=flat_ids.device,
            dtype=torch.float,
        )

        text_mask = token_types == 1
        if text_mask.any():
            text_token_ids = self.text_kgl2token[offsets[text_mask]]
            result[text_mask] = self.retrieve_text(text_token_ids)

        image_mask = token_types == 2
        if image_mask.any():
            result[image_mask] = self.retrieve_image(flat_ids[image_mask]).reshape(
                -1, self.config.r
            )

        return result.reshape(*kgl_ids.shape, -1)


class ContextRetriever(BasePNARetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.up_scaling = nn.Linear(
            self.config.r,
            self.config.llm_hidden_dim,
            bias=False,
            dtype=torch.float,
        )

    def forward(self, kgl_ids):
        kg_embs = super().forward(kgl_ids)
        return self.up_scaling(kg_embs)


class ScoreRetriever(BasePNARetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        cfg_kg = config.kg_encoder
        cfg_base = cfg_kg.base_layer

        base_layer = PNALayer(
            input_dim=cfg_base.input_dim,
            output_dim=cfg_base.output_dim,
            num_relation=cfg_kg.num_relation,
            query_input_dim=cfg_base.query_input_dim,
            aggregate_func=cfg_base.get("aggregate_func", "pna"),
            layer_norm=cfg_base.get("layer_norm", "yes"),
            dependent=cfg_base.get("dependent", "yes"),
        )

        self.kg_retriever = ConditionedPNA(
            base_layer=base_layer,
            num_layer=cfg_kg.get("num_layer", 6),
            num_mlp_layer=cfg_kg.get("num_mlp_layer", 2),
            node_ratio=cfg_kg.get("node_ratio", 0.1),
            degree_ratio=cfg_kg.get("degree_ratio", 1),
            remove_one_hop=cfg_kg.get("remove_one_hop", "yes"),
        )

        self.h_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.h_image_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.node_fusion_gate = nn.Linear(
            self.config.r * 2,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )
        self.query_fusion_gate = nn.Linear(
            self.config.r * 2,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )

    def fuse_modalities(self, text_embs, image_embs, image_mask, gate_layer):
        image_mask = image_mask.to(text_embs.device).unsqueeze(-1).type_as(text_embs)
        image_embs = image_embs * image_mask
        gate = torch.sigmoid(gate_layer(torch.cat([text_embs, image_embs], dim=-1)))
        fused = text_embs + gate * image_embs
        return self.norm(fused)

    def forward(
        self,
        h_id,
        r_id,
        t_id,
        text_hidden_states,
        image_hidden_states,
        image_kgl_ids,
        rel_hidden_states,
        graph,
        all_index,
        all_text_kgl_index,
        all_image_kgl_index,
    ):
        score_text_embs = super().forward(all_text_kgl_index)
        score_image_embs = super().forward(all_image_kgl_index)
        node_image_mask = self.image_token_has_feature(all_image_kgl_index)
        score_node_embs = self.fuse_modalities(
            score_text_embs,
            score_image_embs,
            node_image_mask,
            self.node_fusion_gate,
        )

        head_text_embeds = self.h_down_scaling(text_hidden_states)
        head_image_embeds = self.h_image_down_scaling(image_hidden_states)
        head_image_mask = self.image_token_has_feature(image_kgl_ids)
        head_embeds = self.fuse_modalities(
            head_text_embeds,
            head_image_embeds,
            head_image_mask,
            self.query_fusion_gate,
        )
        rel_embeds = self.r_down_scaling(rel_hidden_states)

        score = self.kg_retriever(
            h_id,
            r_id,
            t_id,
            head_embeds,
            rel_embeds,
            graph,
            score_node_embs,
            all_index,
        )

        return score


class RelScoreRetriever(BasePNARetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float,
        )

    def forward(self, rel_hidden_states, all_rel_kgl_index):
        score_text_embs = super().forward(all_rel_kgl_index)
        rel_embeds = self.r_down_scaling(rel_hidden_states)
        return F.linear(rel_embeds, score_text_embs)
