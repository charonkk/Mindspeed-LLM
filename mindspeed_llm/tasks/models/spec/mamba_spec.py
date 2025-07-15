# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from mindspeed_llm.core import PTNorm


layer_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                norm=PTNorm,
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=ColumnParallelLinear,
                        out_proj=RowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=PTNorm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=PTNorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)