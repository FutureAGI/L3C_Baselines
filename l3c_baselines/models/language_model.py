import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from l3c_baselines.modules import MLPEncoder, ResidualMLPDecoder, CausalBlock
from l3c_baselines.utils import format_cache
from l3c_baselines.utils import ce_loss_mask, mse_loss_mask
from l3c_baselines.utils import count_parameters
from restools.logging import log_fatal

class LanguageModel(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config, verbose=False):
        super().__init__()

        # 创建动作编码层
        self.word_emb = MLPEncoder(config.word_embeddings)

        self.nvocab = config.vocab_size

        self.encoder = CausalBlock(config.causal_block)

        self.output_mapping = ResidualMLPDecoder(config.output_layers)

        loss_weight = torch.cat((
                torch.linspace(0.0, 1.0, config.context_warmup).unsqueeze(0),
                torch.full((1, config.max_position - config.context_warmup,), 1.0)), dim=1)
        self.register_buffer('loss_weight', loss_weight)

        if(verbose):
            print("Language Model initialized, total params: {}".format(count_parameters(self)))



    def forward(self, inputs, cache=None, need_cache=True, T=1.0, update_memory=True):
        """
        Input Size:
            inputs:[B, NT], int
        """
        outputs = self.word_emb(inputs)

        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        outputs = self.output_mapping(outputs, T=T)

        return outputs, new_cache
    
    def reset(self):
        self.encoder.reset()

    def perplexity(self, inputs, outputs, start_position=0, use_loss_weight=True, update_memory=True):
        seq_len = inputs.shape[1]
        logits, _ = self.forward(inputs, need_cache=False, update_memory=update_memory)
        loss_weight = (outputs.lt(self.nvocab)) * (outputs.ge(0))
        if(self.loss_weight.shape[1] < start_position + seq_len):
            log_fatal(f"specified max_position {self.loss_weight.shape[1]} is shorter" +
                    f" than sequence length {start_position + seq_len}, quit")
        if(use_loss_weight):
            loss_weight = self.loss_weight[:, start_position:(start_position + seq_len)] * loss_weight
        return ce_loss_mask(logits, outputs, gamma=0, mask=loss_weight)

    def perplexity_array(self, inputs, outputs, start_position=0, use_loss_weight=True, update_memory=True):
        seq_len = inputs.shape[1]
        logits, _ = self.forward(inputs, need_cache=False, update_memory=update_memory)
        loss_weight = (outputs.lt(self.nvocab)) * (outputs.ge(0))
        if(use_loss_weight):
            loss_weight = self.loss_weight[:, start_position:(start_position + seq_len)] * loss_weight
        return ce_loss_mask(logits, outputs, gamma=0, mask=loss_weight, reduce=None)

    def inference_seg(self, inputs, L, 
                      temp_default=1.0, 
                      temp_setting=None, 
                      cache=None):
        with torch.no_grad():
            sampled_outputs = inputs
            outputs = inputs
            T = temp_default
            for _ in range(L):
                logits, cache = self.forward(sampled_outputs, cache=cache, need_cache=True, T=T)
                sampled_outputs = torch.multinomial(logits, num_samples=1)
                #sampled_outputs = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                outputs = torch.cat([outputs, sampled_outputs], dim=-1)
                if(temp_setting is not None):
                    assert sampled_outputs.shape[0] == 1, "T_setting is only for batch_size=1"
                    token = sampled_outputs[0][-1].item()
                    if token in temp_setting:
                        T = temp_setting[token]
                    else:
                        T = temp_default
        return outputs


if __name__=='__main__':
    import sys
    from l3c_baselines.utils import Configure
    config = Configure()
    config.from_yaml(sys.argv[1])
    LM = LanguageModel(config.model_config)
    cache = None
    sp = 0
    for seg in range(5):
        inputs = torch.randint(0, 128, (4, 64))
        outputs, cache = LM.forward(inputs, cache=cache)
        print(seg, outputs.shape)
        print(format_cache(cache, "Cache"))
        print(format_cache(LM.encoder.layers.memory, "Memory1"))
        ppl = LM.perplexity(inputs[:, :-1], inputs[:, 1:], start_position=sp, update_memory=False)
        print(format_cache(LM.encoder.layers.memory, "Memory2"))
        sp += seg
        print(ppl)
        print(ppl.shape)
