"""CPU equivalents of Falcon's masked attention and interleaved squared-ReLU gate.

Uses ordinary PyTorch operations, bounded query chunks, and the original model weights.
No Triton kernels, torch.compile, or CUDA fallback.
"""

import copy
from types import SimpleNamespace

import torch


class CPUMask:
    BLOCK_SIZE = (128, 128)

    def __init__(self, mask_mod):
        self.mask_mod = mask_mod
        self.seq_lengths = None

    def __getitem__(self, _key):
        return copy.copy(self)


def offset_mask_mod(mask_mod, offset):
    return lambda b, h, q, kv: mask_mod(b, h, q + offset, kv)


def create_batch_attention_mask(
    input_batch, *, pad_token_id, eos_token_id, soi_token_id, eoi_token_id, max_len=None
):
    eos = input_batch == eos_token_id
    eos[:, -1] = True
    cumulative = eos.long().cumsum(1)
    documents = torch.zeros_like(cumulative)
    documents[:, 1:] = cumulative[:, :-1]
    non_left_pad = (input_batch != pad_token_id).long().cumsum(1) > 0
    starts = (input_batch == soi_token_id).long().cumsum(1)
    ends = (input_batch == eoi_token_id).long().cumsum(1)
    in_image = starts - ends > 0
    image_ids = starts * in_image

    def mask(b, _h, q, kv):
        causal = (q >= kv) & (documents[b, q] == documents[b, kv]) & non_left_pad[b, kv]
        same_image = in_image[b, q] & in_image[b, kv] & (image_ids[b, q] == image_ids[b, kv])
        return causal | same_image

    return CPUMask(mask)


def cpu_attention(q, k, v, *, block_mask, return_aux=None, **_kwargs):
    outputs, log_sums = [], []
    batch = torch.arange(q.shape[0]).view(-1, 1, 1, 1)
    keys = torch.arange(k.shape[-2]).view(1, 1, 1, -1)
    transposed = k.transpose(-2, -1)
    for start in range(0, q.shape[-2], 64):
        query = q[:, :, start : start + 64]
        positions = torch.arange(start, start + query.shape[-2]).view(1, 1, -1, 1)
        allowed = block_mask.mask_mod(batch, 0, positions, keys)
        scores = torch.matmul(query, transposed) * (q.shape[-1] ** -0.5)
        scores.masked_fill_(~allowed, float("-inf"))
        lse = scores.logsumexp(-1)
        probabilities = scores.softmax(-1).nan_to_num(0)
        outputs.append(torch.matmul(probabilities, v))
        log_sums.append(lse)
    output = torch.cat(outputs, dim=-2)
    if return_aux is not None:
        return output, SimpleNamespace(lse=torch.cat(log_sums, dim=-1))
    return output


compiled_flex_attn_decode = cpu_attention
compiled_flex_attn_prefill = cpu_attention
