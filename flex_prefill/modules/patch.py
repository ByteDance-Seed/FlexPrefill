# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import types
import warnings
import functools

import torch
from transformers import PreTrainedModel

from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention

try:
    import vllm
    from vllm import _custom_ops as ops
    from vllm.attention.backends.abstract import AttentionType
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
except:
    pass


ATTENTION_CONFIG_EXAMPLE = {
    "default": {},
    "flash": {},
    "streaming_llm": {
        "global_window": 1024,
        "local_window": 8192,
    },
    "minfer": {
        "model_type": "llama3.1",
    },
    "vertical_slash": {
        "block_size": 128,
        "vertical_size": 8192,
        "slash_size": 8192,
    },
    "flex_prefill": {
        "block_size": 128,
        "flex_prefill_gamma": 0.9,
        "flex_prefill_tau": 0.1,
        "flex_prefill_min_budget": 1024,
        "flex_prefill_max_budget": None,
    },
}


def get_config_example(pattern: str = None):
    if pattern is None:
        return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE)
    else:
        return copy.deepcopy(ATTENTION_CONFIG_EXAMPLE[pattern])


def patch_model_config(model: PreTrainedModel, pattern: str, cfg: dict):
    if pattern == "minfer":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if cfg["model_type"].lower() == "llama3.1":
            cfg_path = os.path.abspath(
                os.path.join(
                    current_dir,
                    "../ops/minfer/config/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json",
                )
            )
        elif cfg["model_type"].lower() == "glm4":
            cfg_path = os.path.abspath(
                os.path.join(
                    current_dir,
                    "../ops/minfer/config/GLM_4_9B_1M_instruct_kv_out_v32_fit_o_best_pattern.json",
                )
            )
        elif cfg["model_type"].lower() == "qwen2":
            cfg_path = os.path.abspath(
                os.path.join(
                    current_dir,
                    "../ops/minfer/config/Qwen2_7B_Instruct_128k_instruct_kv_out_v32_fit_o_best_pattern.json",
                )
            )
        elif cfg["model_type"].lower() == "yi":
            cfg_path = os.path.abspath(
                os.path.join(
                    current_dir,
                    "../ops/minfer/config/Yi_9B_200k_kv_out_v32_fit_o_best_pattern.json",
                )
            )
        else:
            raise ValueError(f"unknown minfer model type {cfg['model_type']}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        setattr(model.config, "minfer_config", cfg)
    else:
        for k, v in ATTENTION_CONFIG_EXAMPLE[pattern].items():
            if k not in cfg:
                cfg[k] = v
        for k, v in cfg.items():
            setattr(model.config, k, v)


def patch_llama_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    from transformers.models.llama.modeling_llama import (
        LlamaFlashAttention2,
        LlamaForCausalLM,
        LlamaMLP,
    )

    assert isinstance(model, LlamaForCausalLM)

    if pattern == "flash":
        from flex_prefill.modules.llama.flash_attention import (
            llama_flash_attention_forward,
        )

        new_forward_func = llama_flash_attention_forward
    elif pattern == "minfer":
        from flex_prefill.modules.llama.minfer_attention import (
            llama_minfer_attention_forward,
        )

        new_forward_func = llama_minfer_attention_forward
    elif pattern == "vertical_slash":
        from flex_prefill.modules.llama.vertical_slash_attention import (
            llama_vertical_slash_attention_forward,
        )

        new_forward_func = llama_vertical_slash_attention_forward
    elif pattern == "streaming_llm":
        from flex_prefill.modules.llama.streaming_llm_attention import (
            llama_streaming_llm_attention_forward,
        )

        new_forward_func = llama_streaming_llm_attention_forward
    elif pattern == "flex_prefill":
        from flex_prefill.modules.llama.flex_prefill_attention import (
            llama_flex_prefill_attention_forward,
        )

        new_forward_func = llama_flex_prefill_attention_forward
    else:
        raise ValueError(f"unknown attention pattern {pattern}")

    for idx, m in model.named_modules():
        if isinstance(m, LlamaFlashAttention2):
            m.forward = types.MethodType(new_forward_func, m)

    from flex_prefill.modules.llama.causal_model_forward import (
        llama_causal_model_forward,
    )

    # multiple gpus inference using Accelerate
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(llama_causal_model_forward, model)
    else:
        model.forward = types.MethodType(llama_causal_model_forward, model)

    from flex_prefill.modules.llama.llama_mlp_forward import llama_mlp_forward

    for idx, m in model.named_modules():
        if isinstance(m, LlamaMLP):
            m.forward = types.MethodType(llama_mlp_forward, m)


def patch_glm_attention(model: PreTrainedModel, pattern):
    if pattern == "default":
        return

    assert type(model).__name__ == "ChatGLMForConditionalGeneration"

    if pattern == "flash":
        from flex_prefill.modules.glm.flash_attention import (
            glm_flash_attention_forward,
        )

        new_forward_func = glm_flash_attention_forward
    elif pattern == "streaming_llm":
        from flex_prefill.modules.glm.streaming_llm_attention import (
            glm_streaming_llm_attention_forward,
        )

        new_forward_func = glm_streaming_llm_attention_forward
    elif pattern == "minfer":
        from flex_prefill.modules.glm.minfer_attention import (
            glm_minfer_attention_forward,
        )

        new_forward_func = glm_minfer_attention_forward
    elif pattern == "vertical_slash":
        from flex_prefill.modules.glm.vertical_slash_attention import (
            glm_vertical_slash_attention_forward,
        )

        new_forward_func = glm_vertical_slash_attention_forward
    elif pattern == "flex_prefill":
        from flex_prefill.modules.glm.flex_prefill_attention import (
            glm_flex_prefill_attention_forward,
        )

        new_forward_func = glm_flex_prefill_attention_forward
    else:
        raise ValueError(f"unknown attention pattern {pattern}")
    from flex_prefill.modules.glm.glm_mlp_forward import glm_mlp_forward
    from flex_prefill.modules.glm.glm_self_attention_foward import (
        glm_self_attention_forward,
    )

    for idx, m in model.named_modules():
        if type(m).__name__ == "FlashAttention2":
            m.forward = types.MethodType(new_forward_func, m)
        if type(m).__name__ == "SelfAttention":
            m.forward = types.MethodType(glm_self_attention_forward, m)
        if type(m).__name__ == "MLP":
            m.forward = types.MethodType(glm_mlp_forward, m)


def patch_qwen2_attention(model: PreTrainedModel, pattern: str):
    if pattern == "default":
        return

    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2FlashAttention2,
        Qwen2ForCausalLM,
        Qwen2MLP,
    )

    assert isinstance(model, Qwen2ForCausalLM)

    if pattern == "flash":
        from flex_prefill.modules.qwen2.flash_attention import (
            qwen2_flash_attention_forward,
        )

        new_forward_func = qwen2_flash_attention_forward
    elif pattern == "streaming_llm":
        from flex_prefill.modules.qwen2.streaming_llm_attention import (
            qwen2_streaming_llm_attention_forward,
        )

        new_forward_func = qwen2_streaming_llm_attention_forward
    elif pattern == "minfer":
        from flex_prefill.modules.qwen2.minfer_attention import (
            qwen2_minfer_attention_forward,
        )

        new_forward_func = qwen2_minfer_attention_forward
    elif pattern == "vertical_slash":
        from flex_prefill.modules.qwen2.vertical_slash_attention import (
            qwen2_vertical_slash_attention_forward,
        )

        new_forward_func = qwen2_vertical_slash_attention_forward
    elif pattern == "flex_prefill":
        from flex_prefill.modules.qwen2.flex_prefill_attention import (
            qwen2_flex_prefill_attention_forward,
        )

        new_forward_func = qwen2_flex_prefill_attention_forward
    else:
        raise ValueError(f"unknown attention pattern {pattern}")

    from flex_prefill.modules.qwen2.qwen_mlp_forward import qwen2_mlp_forward

    for idx, m in model.named_modules():
        if isinstance(m, Qwen2FlashAttention2):
            m.forward = types.MethodType(new_forward_func, m)
        if isinstance(m, Qwen2MLP):
            m.forward = types.MethodType(qwen2_mlp_forward, m)

    from flex_prefill.modules.qwen2.causal_model_forward import (
        qwen2_causal_model_forward,
    )

    # multiple gpus inference using Accelerate
    if isinstance(model.forward, functools.partial):
        model.forward.__wrapped__ = types.MethodType(qwen2_causal_model_forward, model)
    else:
        model.forward = types.MethodType(qwen2_causal_model_forward, model)


def disable_hf_flash_attention_check():
    from transformers import PreTrainedModel

    def _enable_flash_attn_2(
        config,
        torch_dtype=None,
        device_map=None,
        check_device_map=True,
        hard_check_only=False,
    ):
        config._attn_implementation = "flash_attention_2"
        return config

    PreTrainedModel._check_and_enable_flash_attn_2 = _enable_flash_attn_2


def patch_hf_model(model: PreTrainedModel, pattern: str, cfg: dict):
    model_name = type(model).__name__
    if "llama" in model_name.lower():
        patch_llama_attention(model, pattern)
    elif "glm" in model_name.lower():
        patch_glm_attention(model, pattern)
    elif "qwen2" in model_name.lower():
        patch_qwen2_attention(model, pattern)
    else:
        raise ValueError(f"unsupported model type: {model_name}")

    patch_model_config(model, pattern, cfg)

    message = f"# replace flash attention with {pattern} attention #"
    print("#" * len(message))
    print(message)
    print("#" * len(message))


def patch_vllm_model(model, pattern: str, cfg: dict):
    assert isinstance(model, vllm.LLM)
    assert pattern in [
        "default",
        "flash",
        "flex_prefill",
    ], "only support default, flash and flex_prefill"

    if pattern != "flex_prefill":
        return

    model.llm_engine.scheduler_config.chunked_prefill_enabled = False
    model.llm_engine.scheduler_config.max_num_seqs = 1
    warnings.warn(
        "flex_prefill will set max_num_seqs=1 and chunked_prefill_enabled=False"
    )

    def flex_prefill_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttentionImpl"
            )

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert (
            k_scale == 1.0 and v_scale == 1.0
        ), "key/v_scale is not supported in FlashAttention."

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (
                kv_cache is None
                or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0
            ):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                # out = torch.ops.vllm.flash_attn_varlen_func(
                #     q=query,
                #     k=key,
                #     v=value,
                #     cu_seqlens_q=prefill_meta.seq_start_loc,
                #     cu_seqlens_k=prefill_meta.seq_start_loc,
                #     max_seqlen_q=prefill_meta.max_prefill_seq_len,
                #     max_seqlen_k=prefill_meta.max_prefill_seq_len,
                #     softmax_scale=self.scale,
                #     causal=True,
                #     window_size=self.sliding_window,
                #     alibi_slopes=self.alibi_slopes,
                #     softcap=self.logits_soft_cap,
                # )
                out = flex_prefill_attention(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    gamma=cfg["flex_prefill_gamma"],
                    tau=cfg.get("flex_prefill_tau", 0),
                    min_budget=cfg.get("flex_prefill_min_budget", None),
                    max_budget=cfg.get("flex_prefill_max_budget", None),
                ).squeeze(0)
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:num_prefill_tokens] = (
                    torch.ops.vllm.flash_attn_varlen_func(  # noqa
                        q=query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_k=max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        softcap=self.logits_soft_cap,
                    )
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = torch.ops.vllm.flash_attn_with_kvcache(
                decode_query.unsqueeze(1),
                key_cache,
                value_cache,
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            ).squeeze(1)

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    for (
        idx,
        m,
    ) in (
        model.llm_engine.model_executor.driver_worker.model_runner.model.named_modules()
    ):
        if "attention" in type(m).__name__.lower() and hasattr(m, "impl"):
            m.impl.forward = types.MethodType(flex_prefill_forward, m.impl)


def patch_model(model, pattern: str, cfg: dict):
    """Patch the attention mechanism of a transformers model or vllm model to use a specified pattern.

    Args:
        model: The model to be patched. It must be either a `PreTrainedModel` from the Hugging Face transformers
            library or a vllm model.
        pattern (str): The attention pattern to apply to the model. Supported patterns include:
            - 'default': Default attention mechanism with no additional configuration.
            - 'flash': Flash attention mechanism with no additional configuration.
            - 'streaming_llm': Streaming LLM attention with init tokens and local window.
            - 'minfer': Minference attention with model-specific settings.
            - 'flex_prefill': FlexPrefill attention.
        cfg (dict): Configuration settings for the specified pattern. The required keys and values depend on
            the chosen pattern. For example:
            - 'streaming_llm': {'global_window': 1024, 'local_window': 8192}
            - 'minfer' : {'model_type': 'llama3.1'}, supported model_type: llama3.1, qwen2, yi, glm4
            - 'flex_prefill': {'block_size': 128, 'flex_prefill_gamma': 0.9, 'flex_prefill_tau': 0.1,
                              'flex_prefill_min_budget': 512, 'flex_prefill_max_budget': 32768}
            Refer to `flex_prefill.modules.patch.ATTENTION_CONFIG_EXAMPLE` for full details on supported patterns and their configurations.
    """
    if isinstance(model, PreTrainedModel):
        patch_hf_model(model, pattern, cfg)
    else:
        patch_vllm_model(model, pattern, cfg)
    return model
