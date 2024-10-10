import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from l3c_baselines.modules import EncodeBlock, DecodeBlock, CausalBlock

class LanguageModel(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()

        # 创建动作编码层
        self.word_encoder = EncodeBlock(config.word_embeddings)

        self.encoder = CausalBlock(config.causal_block)

        self.output_mapping = nn.Sequential(nn.Linear(d_model, vocab_size), nn.Softmax(dim=-1))

    def forward(self, inputs, cache=None, need_cache=True):
        """
        Input Size:
            inputs:[B, NT], int
        """
        B, NT = inputs.shape

        #calculate cached positions
        if(cache is None):
            cache_len = 0
        else:
            assert isinstance(cache, list) and len(cache) == self.num_layers + 1, "The cache must be list with length == num_layers + 1"
            cache_len = cache[0].shape[1]

        # Input actions: [B, NT, 1, H]
        outputs = self.word_embedding(inputs)

        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache, checkpoints_density=self.checkpoints_density)

        outputs = self.output_mapping(self.norm(outputs))

        return outputs, new_cache

if __name__=='__main__':
    inputs = torch.randint(0, 1024, (4, 64))
    ART2 = LanguageModel(1024, 8, 128, 8, 1024, checkpoints_density=4)
    out_nlp, cache = ART2(inputs, need_cache=True)
    out_nlp2, cache = ART2(inputs, cache=cache, need_cache=True)
    print(out_nlp.shape, out_nlp2.shape)