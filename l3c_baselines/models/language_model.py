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
        self.word_emb = EncodeBlock(config.word_embeddings)

        self.nvocab = config.vocab_size

        self.encoder = CausalBlock(config.causal_block)

        self.output_mapping = DecodeBlock(config.output_layers)

        loss_weight = torch.cat((
                torch.linspace(0.0, 1.0, config.context_warmup).unsqueeze(0),
                torch.full((1, config.max_time_step - config.context_warmup,), 1.0)), dim=1)
        self.register_buffer('loss_weight', loss_weight)


    def forward(self, inputs, cache=None, need_cache=True, T=1.0):
        """
        Input Size:
            inputs:[B, NT], int
        """
        outputs = self.word_emb(inputs)

        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache)

        outputs = self.output_mapping(outputs, T=T)

        return outputs, new_cache
    
    def reset(self):
        self.encoder.reset()

    def perplexity(self, inputs, outputs, start_position=0):
        seq_len = inputs.shape[1]
        logits, new_cache = self.forward(inputs, need_cache=False, use_loss_weight=True)
        loss_weight = (logits.lt(self.nvocab)) * (logits.ge(0))
        if(use_loss_weight):
            loss_weight = self.loss_weight[:, start_position:(start_position + seq_len)] * loss_weight
        return ce_loss_mask(logits, outputs, gamma=0, mask=loss_weight)

    def perplexity_array(self, inputs, outputs, start_position=0, use_loss_weight=True):
        seq_len = inputs.shape[1]
        logits, new_cache = self.forward(inputs, need_cache=False)
        loss_weight = (logits.lt(self.nvocab)) * (logits.ge(0))
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
    inputs = torch.randint(0, 1024, (4, 64))
    ART2 = LanguageModel(1024, 8, 128, 8, 1024, checkpoints_density=4)
    out_nlp, cache = ART2(inputs, need_cache=True)
    out_nlp2, cache = ART2(inputs, cache=cache, need_cache=True)
    print(out_nlp.shape, out_nlp2.shape)