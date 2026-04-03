# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import logging
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, mpu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import cal_split_sizes, split_forward_gather_backward, gather_forward_split_backward
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses, 
    get_context_parallel_group_for_hybrid_ring, 
    get_context_parallel_for_hybrid_ulysses_world_size
)
from mindspeed.utils import set_actual_seq_len
from mindspeed_mm.models.common.embeddings.rope import DynamicRotaryEmbedding
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2VLRotaryEmbedding_llm
from mindspeed_mm.models.vision.vision_encoders.qwen3vl_vit_model import Qwen3VLTextRotaryEmbedding_llm
from mindspeed_mm.models.text_decoder.qwen3vl_transformer_block import Qwen3vlTransformerBlock
from mindspeed_mm.utils.utils import ensure_valid, split_forward_gather_backward_with_megatron_cp
from mindspeed_mm.models.vision.vision_encoders.glm4v_vl_vit_model import GlmTransformerBlock, Glm4vRotaryEmbedding_llm


class MMGPTModel(LanguageModule):
    """MMGPTModel Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        reward_process (bool, optional): Without an output layer (only used with videoalign). Defaults to False.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        reward_process: bool = False,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['mrope', 'rope'] = 'mrope',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.reward_process = reward_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'mrope':
            if getattr(config, 'mrope_section', None) is None and getattr(config, 'rope_scaling', None) is None:
                raise AssertionError('mrope section should be provided for mrope!')
            if getattr(config, 'model_id', None) == "glm4v_lm":
                self.rotary_pos_emb = Glm4vRotaryEmbedding_llm(config=config)
            elif getattr(config, 'model_id', None) == "qwen3_lm":
                self.rotary_pos_emb = Qwen3VLTextRotaryEmbedding_llm(config=config)
            else:
                self.rotary_pos_emb = Qwen2VLRotaryEmbedding_llm(config=config)
        elif self.position_embedding_type == 'rope':
            if getattr(self.config, "rope_scaling", None) is None:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                )
            elif self.config.rope_scaling.type == 'dynamic':
                self.rotary_pos_emb = DynamicRotaryEmbedding(
                    config=self.config,
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                )
            else:
                raise AssertionError(f'Unsupported rope scaling type: {self.config.rope_scaling.type}')
        # Transformer.
        if getattr(config, 'model_id', None) == "glm4v_lm":
            self.decoder = GlmTransformerBlock(
                config=self.config,
                spec=transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        elif getattr(config, 'model_id', None) == "qwen3_lm":
            self.decoder = Qwen3vlTransformerBlock(
                config=self.config,
                spec=transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        else:
            self.decoder = TransformerBlock(
                config=self.config,
                spec=transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError('input_tensor should only be length 1 for gpt/bert')
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        output_hidden_states: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """GPT 模型的前向传播函数：依次经过 embedding、decoder 和可选的输出层。
        若提供 labels 则返回损失，否则返回最终隐藏状态或 logits。
        """
        # 1. 准备 decoder 输入：若外部已提供 decoder_input 则直接使用；否则由 embedding 层生成。
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # 非首段 pipeline stage，decoder 输入由上游传入
            decoder_input = None

        # 2. 若开启 remove_padding，先根据 position_ids 计算实际序列长度，供后续算子使用
        if getattr(self.config, 'use_remove_padding', False):
            if position_ids is not None and position_ids.dim() == 3:
                position_ids_fa = position_ids[0]
            position_ids_fa = position_ids_fa.flatten()
            indices_q = torch.arange(position_ids_fa.size(0), device=position_ids_fa.device, dtype=torch.int32)
            cu_seqlens = torch.cat(
                (
                    indices_q[position_ids_fa == 0],
                    torch.tensor(position_ids_fa.size(), device=position_ids_fa.device, dtype=torch.int32),
                )
            )
            set_actual_seq_len(tuple(cu_seqlens[1:].cpu().numpy().tolist()))

        # 3. 若启用 Context Parallel（CP），按算法将输入拆分到不同 rank
        if mpu.get_context_parallel_world_size() > 1:
            split_gather_sizes = cal_split_sizes(input_ids.shape[-1], mpu.get_context_parallel_world_size())

            if get_args().context_parallel_algo == "ulysses_cp_algo":
                # Ulysses CP：按 seq 维度切分
                input_ids = split_forward_gather_backward(input_ids, mpu.get_context_parallel_group(), 1,
                                                            split_gather_sizes, "down")
                position_ids = split_forward_gather_backward(position_ids, mpu.get_context_parallel_group(), 2,
                                                            split_gather_sizes, "down")
                if self.pre_process:
                    decoder_input = split_forward_gather_backward(decoder_input, mpu.get_context_parallel_group(), 0,
                                                                split_gather_sizes, "down")
                    
            elif get_args().context_parallel_algo == "megatron_cp_algo":
                # Megatron CP：按指定 dim 切分
                input_ids = split_forward_gather_backward_with_megatron_cp(input_ids, mpu.get_context_parallel_group(), dim=1)
                if position_ids is not None:
                    position_ids = split_forward_gather_backward_with_megatron_cp(position_ids, mpu.get_context_parallel_group(), dim=2)
                if self.pre_process:
                    decoder_input = split_forward_gather_backward_with_megatron_cp(decoder_input, mpu.get_context_parallel_group(), dim=0)

            elif get_args().context_parallel_algo == "hybrid_cp_algo":
                # 混合 CP：先 Ulysses 再 Megatron
                seq_len = input_ids.shape[-1]
                split_gather_sizes = cal_split_sizes(seq_len, get_context_parallel_for_hybrid_ulysses_world_size())

                input_ids = split_forward_gather_backward(input_ids, get_context_parallel_group_for_hybrid_ulysses(), 1, split_gather_sizes, "down")
                input_ids = split_forward_gather_backward_with_megatron_cp(input_ids, get_context_parallel_group_for_hybrid_ring(), dim=1)

                if position_ids is not None:
                    position_ids = split_forward_gather_backward(position_ids, get_context_parallel_group_for_hybrid_ulysses(), 2, split_gather_sizes, "down")
                    position_ids = split_forward_gather_backward_with_megatron_cp(position_ids, get_context_parallel_group_for_hybrid_ring(), dim=2)

                if self.pre_process:
                    decoder_input = split_forward_gather_backward(decoder_input, get_context_parallel_group_for_hybrid_ulysses(), 0, split_gather_sizes, "down")
                    decoder_input = split_forward_gather_backward_with_megatron_cp(decoder_input, get_context_parallel_group_for_hybrid_ring(), dim=0)

        # 4. 若开启 Sequence Parallel（SP），在 CP 之后将 decoder_input 打散到 TP 组
        if self.config.sequence_parallel and self.pre_process:
            decoder_input = scatter_to_sequence_parallel_region(decoder_input)

        # 5. 生成旋转位置编码（RoPE / mRoPE）
        rotary_pos_emb = None
        if self.position_embedding_type == 'mrope':
            param_dtype = torch.bfloat16
            if not getattr(self.config, 'bf16', False):
                raise AssertionError('mrope only support bf16 now!')
            if getattr(self.config, 'model_id', None) == "glm4v_lm":
                rotary_pos_emb = self.rotary_pos_emb(input_ids.device, param_dtype, position_ids, self.config.mrope_section)
            elif getattr(self.config, 'model_id', None) == "qwen3_lm":
                rotary_pos_emb = self.rotary_pos_emb(input_ids.device, param_dtype, position_ids)
                half_dim = rotary_pos_emb.shape[-1] // 2
                cos, sin = rotary_pos_emb[..., :half_dim], rotary_pos_emb[..., half_dim:]
                rotary_pos_emb = torch.cat([cos, sin], dim=0)
            else:
                # qwen2vl 默认分支：按 mrope_section 重组 cos/sin
                rotary_pos_emb = self.rotary_pos_emb(input_ids.device, param_dtype, position_ids)
                half_dim = rotary_pos_emb.shape[-1] // 2
                cos, sin = rotary_pos_emb[..., :half_dim], rotary_pos_emb[..., half_dim:]
                mrope_section = self.config.mrope_section * 2
                cos = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(2)
                sin = torch.cat([m[:, i % 3, :, :] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(2)
                rotary_pos_emb = torch.cat([cos, sin], dim=0)
        elif self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, self.decoder, decoder_input, self.config, inference_params
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # 6. 经过 decoder 层计算隐藏状态
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        # 7. 若非最后阶段或仅 reward 模式，直接返回隐藏状态
        if not self.post_process or self.reward_process:
            return hidden_states

        # 8. 输出层计算 logits；若提供 labels 则计算损失
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # 推理/生成阶段：转置后返回 logits；按需返回隐藏状态
            logits = logits.transpose(0, 1).contiguous()
            if output_hidden_states:
                return {"logits": logits, "hidden_states": hidden_states}
            return logits

        # 训练阶段：计算语言模型损失
        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        ensure_valid(not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}')

        return sharded_state_dict
