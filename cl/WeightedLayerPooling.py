from __future__ import annotations
import json
import os
import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import Tensor, nn


class WeightedLayerPooling(nn.Module):
    """Token embeddings are weighted mean of their different hidden layer representations"""

    def __init__(
        self, word_embedding_dimension, num_hidden_layers, layer_start, layer_weights=None
    ):
        super().__init__()
        self.config_keys = ["word_embedding_dimension", "layer_start", "num_hidden_layers"]
        self.word_embedding_dimension = word_embedding_dimension
        self.layer_start = min(layer_start, num_hidden_layers - 1)
        self.num_hidden_layers = num_hidden_layers
        num_layers_to_use = num_hidden_layers - self.layer_start

        print(f"🔍[WeightedLayerPooling] Using {num_layers_to_use} layers from layer {self.layer_start} to {num_hidden_layers - 1}")

        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers - layer_start), dtype=torch.float))
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        ft_all_layers = features["all_layer_embeddings"]
        all_layer_embedding = torch.stack(ft_all_layers)

        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :] 

        # 确保 self.layer_weights 在正确的设备上
        if self.layer_weights.device != all_layer_embedding.device:
            self.layer_weights = nn.Parameter(self.layer_weights.to(all_layer_embedding.device))

        # 检查 layer_weights 尺寸是否匹配
        if self.layer_weights.size(0) != all_layer_embedding.size(0):
            print(f"Warning: layer_weights size ({self.layer_weights.size(0)}) does not match embeddings ({all_layer_embedding.size(0)})")
            self.layer_weights = nn.Parameter(torch.ones(all_layer_embedding.size(0), dtype=torch.float, device=all_layer_embedding.device))

        # 计算加权平均
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(all_layer_embedding)
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({"token_embeddings": weighted_average})

        # 平均池化
        token_embeddings = features["token_embeddings"]
        attention_mask = (
            features["attention_mask"]
            if "attention_mask" in features
            else torch.ones(token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.int64)
        )

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

        features["sentence_embedding"] = sentence_embedding
        return features


    def get_word_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = WeightedLayerPooling(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return model
    
