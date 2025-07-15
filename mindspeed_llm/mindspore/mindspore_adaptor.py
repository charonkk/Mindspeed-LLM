# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved

from mindspeed_llm.mindspore.utils import clear_wrapper
from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation, MegatronAdaptationABC


class MindSporeAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=True, create_dummy=False, check_patch=False):
        """
        Register adaptations into collection. Force patch for MindSpore patches.
        """
        if check_patch:
            new_func = MindSporeAdaptation.wrap_print_new_func(new_func)
        MegatronAdaptation.register(orig_func_name, new_func, force_patch, create_dummy)

    @classmethod
    def wrap_print_new_func(cls, new_func):
        from functools import wraps

        # wrap the new func with info print
        def make_patch(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Stepping into MindSpore patch: {func.__name__}")
                return func(*args, **kwargs)

            return wrapper

        # wrap new_func before handing it off to MegatronAdaptation.register
        new_func_with_print = make_patch(new_func)
        return new_func_with_print

    def execute(self):
        args = MegatronAdaptation.get_args()
        if not hasattr(args, "ai_framework") or args.ai_framework != "mindspore":
            return

        args = self.reparse_args()

        from mindspeed_llm.mindspore.training.arguments import _validate_optimizer
        MindSporeAdaptation.register('mindspeed_llm.training.arguments._validate_optimizer', _validate_optimizer)

        from ..core.models.gpt.gpt_model import GPTModel
        from ..mindspore.core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward
        from mindspeed.mindspore.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_with_cp
        from mindspeed.mindspore.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward

        MindSporeAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                    distributed_data_parallel_init_with_cp)
        from megatron.core.transformer.moe.moe_layer import MoELayer
        clear_wrapper('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', MoELayer.__init__)
        MindSporeAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                moe_layer_init_wrapper)
        MindSporeAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.forward', moe_layer_forward)

        from .core.transformer.moe.moe_utils import unpermute
        MindSporeAdaptation.register('mindspeed.core.transformer.moe.moe_utils.unpermute', unpermute)

        if args.moe_permutation_async_comm:
            if args.moe_token_dispatcher_type == 'alltoall':
                if args.moe_alltoall_overlap_comm:
                    from mindspeed.mindspore.core.transformer.moe.legacy_a2a_token_dispatcher import alltoall_token_permutation_new, \
                            alltoall_token_unpermutation_new
                    from mindspeed.mindspore.core.transformer.moe.experts import group_mlp_forward
                    MindSporeAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    MindSporeAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    MindSporeAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)

                if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute and not args.moe_expert_capacity_factor:
                    from mindspeed.mindspore.core.fusions.npu_moe_token_permute import permute_wrapper
                    from mindspeed.mindspore.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
                    MindSporeAdaptation.register('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
                    MindSporeAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)

        if not args.moe_alltoall_overlap_comm and not args.moe_fb_overlap:
            MindSporeAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
                                    groupedmlp_forward)

        from mindspeed.mindspore.core.distributed.distributed_data_parallel import distributed_data_parallel_init, local_make_param_hook
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__', distributed_data_parallel_init)
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook', local_make_param_hook)

        from mindspeed.mindspore.core.distributed.param_and_grad_buffer import register_grad_ready
        MindSporeAdaptation.register('megatron.core.distributed.param_and_grad_buffer.register_grad_ready', register_grad_ready)

        from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import get_rotary_seq_len, local_rotate_half
        MindSporeAdaptation.register('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len', get_rotary_seq_len)
        MindSporeAdaptation.register('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

        from mindspeed.mindspore.core.optimizer import get_megatron_optimizer
        MindSporeAdaptation.register('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer)
        from mindspeed.mindspore.core.optimizer.optimizer import megatron_optimizer_init
        MindSporeAdaptation.register('megatron.core.optimizer.optimizer.MegatronOptimizer.__init__', megatron_optimizer_init)

        from mindspeed.mindspore.core.pipeline_parallel.schedules import backward_step, forward_backward_no_pipelining
        from mindspeed_llm.mindspore.core.pipeline_parallel.schedules import forward_step
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_step', forward_step)
        MindSporeAdaptation.register('mindspeed.mindspore.core.pipeline_parallel.schedules.forward_step', forward_step)
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.backward_step', backward_step)
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining', forward_backward_no_pipelining)

        if not args.moe_fb_overlap:
            from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving, forward_backward_pipelining_without_interleaving
            MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving', forward_backward_pipelining_without_interleaving)

        from mindspeed.mindspore.core.tensor_parallel.data import local_build_key_size_numel_dictionaries
        MindSporeAdaptation.register('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries', local_build_key_size_numel_dictionaries) # 1097

        from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward
        MindSporeAdaptation.register('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

        from mindspeed.mindspore.core.tensor_parallel.random import local_set_cuda_rng_state, checkpoint_function_forward, checkpoint_function_backward
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random._set_cuda_rng_state', local_set_cuda_rng_state)
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward', checkpoint_function_forward)
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward', checkpoint_function_backward)

        from ..mindspore.training.utils import get_batch_on_this_tp_rank
        MindSporeAdaptation.register('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

        from ..mindspore.core.tensor_parallel.cross_entropy import calculate_predicted_logits, prepare_gradient_calculation_operands
        MindSporeAdaptation.register(
            'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
            calculate_predicted_logits)
        MindSporeAdaptation.register(
            'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.prepare_gradient_calculation_operands',
            prepare_gradient_calculation_operands)

        from mindspeed.mindspore.core.timers import _get_global_min_max_time
        MindSporeAdaptation.register('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)


        from ..mindspore.core.optimizer.distrib_optimizer import get_parameter_state_dp_zero
        MindSporeAdaptation.register('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
                            get_parameter_state_dp_zero)

        if args.async_log_allreduce:
            from mindspeed.mindspore.core.data_parallel.async_log_allreduce import get_async_reduced_loss_value
            MindSporeAdaptation.register('mindspeed.core.data_parallel.async_log_allreduce.get_async_reduced_loss_value',
                                get_async_reduced_loss_value)

        from mindspeed.mindspore.core.tensor_parallel.random import CheckpointWithoutOutput, \
            CheckpointFunctionWithoutOutput
        MindSporeAdaptation.register('mindspeed.core.tensor_parallel.random.CheckpointWithoutOutput',
                                     CheckpointWithoutOutput)
        MindSporeAdaptation.register('mindspeed.core.tensor_parallel.random.CheckpointFunctionWithoutOutput',
                                     CheckpointFunctionWithoutOutput)

        if args.use_ascend_coc:
            from mindspeed.mindspore.ops.lcal_functional import all_gather_matmul, all_gather_matmul_v2, matmul_reduce_scatter, matmul_all_reduce, pure_matmul
            MindSporeAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.all_gather_matmul', all_gather_matmul)
            MindSporeAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.all_gather_matmul_v2', all_gather_matmul_v2)
            MindSporeAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.matmul_reduce_scatter', matmul_reduce_scatter)
            MindSporeAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.matmul_all_reduce', matmul_all_reduce)
            MindSporeAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.pure_matmul', pure_matmul)

        from mindspeed_llm.mindspore.core.transformer.moe.router import apply_seq_aux_loss, topk_router_gating_func
        MindSporeAdaptation.register('mindspeed_llm.core.transformer.moe.router.apply_seq_aux_loss',
                                    apply_seq_aux_loss)
        MindSporeAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.gating', topk_router_gating_func)

        from mindspeed.mindspore.core.transformer.transformer import core_mlp_forward_wrapper
        from megatron.core.transformer.mlp import MLP
        clear_wrapper('megatron.core.transformer.mlp.MLP.forward', MLP.forward)
        MindSporeAdaptation.register('megatron.core.transformer.mlp.MLP.forward',
            core_mlp_forward_wrapper)



        if args.moe_fb_overlap:
            from mindspeed_llm.mindspore.tasks.models.transformer.multi_head_latent_attention import mla_forward
            MindSporeAdaptation.register('mindspeed_llm.tasks.models.transformer.multi_head_latent_attention.MultiHeadLatentAttention.forward',
                                        mla_forward)

            from mindspeed_llm.mindspore.core.pipeline_parallel.dualpipe.gpt_model import ModelGraph, gpt_model_forward, gpt_model_forward_backward_overlaping
            MindSporeAdaptation.register('mindspeed_llm.core.pipeline_parallel.dualpipe.gpt_model.ModelGraph',
                                        ModelGraph)
            MindSporeAdaptation.register('mindspeed_llm.core.pipeline_parallel.dualpipe.gpt_model.gpt_model_forward',
                                        gpt_model_forward)
            MindSporeAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                        gpt_model_forward_backward_overlaping)
            from mindspeed_llm.mindspore.core.pipeline_parallel.dualpipe.MTP_overlap import mtp_overlap_backward
            MindSporeAdaptation.register('mindspeed_llm.core.pipeline_parallel.dualpipe.MTP_overlap.TransformerMTPoverlap.backward',
                                        mtp_overlap_backward)


            #mindspeed
            
            from mindspeed.mindspore.core.pipeline_parallel.dualpipev.dualpipev_schedules import backward_step_with_model_graph, set_shared_embedding_from_dual_chunk, forward_step_with_model_graph, get_shared_embedding_from_dual_chunk, forward_backward_pipelining_with_cutinhalf
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules.backward_step_with_model_graph',
                                backward_step_with_model_graph)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules.set_shared_embedding_from_dual_chunk',
                                set_shared_embedding_from_dual_chunk)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules.forward_step_with_model_graph',
                                forward_step_with_model_graph)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules.get_shared_embedding_from_dual_chunk',
                                get_shared_embedding_from_dual_chunk)
            MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                forward_backward_pipelining_with_cutinhalf)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.transformer_layer import \
                transformer_layer_recompute
            MindSporeAdaptation.register(
                'mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer.transformer_layer_recompute',
                transformer_layer_recompute)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.transformer_block import transformer_block_forward, transformer_block_forward_backward_overlaping
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.transformer_block.transformer_block_forward',
                                transformer_block_forward)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.transformer_block.transformer_block_forward_backward_overlaping',
                                transformer_block_forward_backward_overlaping)
            
            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.adaptor import _make_param_hook
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.adaptor._make_param_hook',
                                _make_param_hook)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.experts import get_gmm_weight_grad, GroupedMatmulWithWeightGradDetach
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.experts.get_gmm_weight_grad',
                                get_gmm_weight_grad)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.experts.GroupedMatmulWithWeightGradDetach',
                                GroupedMatmulWithWeightGradDetach)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.token_dispatcher import preprocess
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.preprocess',
                                preprocess)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.utils import detach_tensor, run_graph_backward, dummy_forward_step_func, run_graph_forward, NoopLayerGraph, LayerGraph
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.detach_tensor',
                                detach_tensor)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.run_graph_backward',
                                run_graph_backward)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.dummy_forward_step_func',
                                dummy_forward_step_func)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.run_graph_forward',
                                run_graph_forward)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.NoopLayerGraph',
                                NoopLayerGraph)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.utils.LayerGraph',
                                LayerGraph)


            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.overlap_funcs.bwd import transformer_layer_backward_moe, transformer_layer_backward_dense, transformer_layer_backward_noop
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.bwd.transformer_layer_backward_moe',
                                transformer_layer_backward_moe)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.bwd.transformer_layer_backward_dense',
                                transformer_layer_backward_dense)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.bwd.transformer_layer_backward_noop',
                                transformer_layer_backward_noop)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.overlap_funcs.fwd import transformer_layer_forward_moe, transformer_layer_forward_dense, transformer_layer_forward_noop
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwd.transformer_layer_forward_moe',
                                transformer_layer_forward_moe)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwd.transformer_layer_forward_dense',
                                transformer_layer_forward_dense)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwd.transformer_layer_forward_noop',
                                transformer_layer_forward_noop)

            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd import transformer_layer_forward_dense_backward_moe_overlaping, \
                    transformer_layer_forward_moe_backward_dense_overlaping, transformer_layer_forward_dense_backward_dense_overlaping, transformer_layer_forward_moe_backward_moe_overlaping
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_dense_backward_moe_overlaping',
                                transformer_layer_forward_dense_backward_moe_overlaping)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_dense_overlaping',
                                transformer_layer_forward_moe_backward_dense_overlaping)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_dense_backward_dense_overlaping',
                                transformer_layer_forward_dense_backward_dense_overlaping)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_moe_overlaping',
                                transformer_layer_forward_moe_backward_moe_overlaping) 


            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import overlap_matmul
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.weight_grad_store.WeightGradStore.overlap_matmul',
                                overlap_matmul)


            from mindspeed.mindspore.core.transformer.moe.comm_utils import async_all_to_all
            MindSporeAdaptation.register('mindspeed.core.transformer.moe.comm_utils.async_all_to_all',
                        async_all_to_all)
            
            from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.token_dispatcher import alltoall_token_perm1, overlap_stream
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm1',
                alltoall_token_perm1)           
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.overlap_stream',
                overlap_stream)     

            from mindspeed_llm.core.pipeline_parallel.dualpipe.adaptor import dualpipe_forward_step_wrapper
            MindSporeAdaptation.register('mindspeed.mindspore.core.pipeline_parallel.dualpipev.dualpipev_schedules.forward_step_with_model_graph',
                dualpipe_forward_step_wrapper)


        if args.swap_optimizer:
            from mindspeed.mindspore.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2
            MindSporeAdaptation.register('mindspeed.ops.npu_apply_fused_adamw_v2.npu_apply_fused_adamw_v2',
                                npu_apply_fused_adamw_v2)
            from mindspeed.mindspore.core.optimizer.swap_optimizer.swap_optimizer import opt_states_initialization, create_tensor_maps, swap_tensors_to_device, _copy_model_params_to_main_params, swap_adamw_step
            MindSporeAdaptation.register('mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer.opt_states_initialization',
                opt_states_initialization)
            MindSporeAdaptation.register('mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer.create_tensor_maps',
                create_tensor_maps)
            MindSporeAdaptation.register('mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer.swap_tensors_to_device',
                swap_tensors_to_device)
            MindSporeAdaptation.register('mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer._copy_model_params_to_main_params',
                _copy_model_params_to_main_params)
            MindSporeAdaptation.register('mindspeed.optimizer.adamw.AdamW.step', swap_adamw_step)

        if args.enable_share_memory:
            from ..mindspore.tasks.dataset.shared_memory_manager import SharedMemoryManager
            MegatronAdaptation.register(
                'mindspeed_llm.tasks.dataset.shared_memory_manager.SharedMemoryManager', SharedMemoryManager)
            from ..mindspore.training.utils import _compute_actual_seq_len
            MegatronAdaptation.register(
                'mindspeed_llm.training.utils._compute_actual_seq_len', _compute_actual_seq_len)

        from ..mindspore.core.distributed.finalize_model_grads import _update_router_expert_bias
        MindSporeAdaptation.register('mindspeed_llm.core.distributed.finalize_model_grads._update_router_expert_bias', _update_router_expert_bias)

        from ..mindspore.core.transformer.module import set_is_first_microbatch
        MindSporeAdaptation.register('megatron.core.transformer.module.MegatronModule.set_is_first_microbatch', set_is_first_microbatch)

        if args.moe_zerc:
            from mindspeed.mindspore.core.transformer.moe.moe_zerc.fwdbwd import transformer_layer_forward_moe_backward_dense_overlaping_zerc, transformer_layer_forward_moe_backward_moe_overlaping_zerc
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_dense_overlaping',
                        transformer_layer_forward_moe_backward_dense_overlaping_zerc)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_moe_overlaping',
                        transformer_layer_forward_moe_backward_moe_overlaping_zerc)
            from mindspeed.mindspore.core.transformer.moe.moe_zerc.token_dispatcher import zerc_alltoall_token_perm1, zerc_alltoall_token_perm2, zerc_alltoall_token_unperm1, zerc_alltoall_token_unperm2
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm1',
                        zerc_alltoall_token_perm1)

            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm2',
                        zerc_alltoall_token_perm2)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm1',
                        zerc_alltoall_token_unperm1)
            MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm2',
                        zerc_alltoall_token_unperm2)

        from mindspeed.mindspore.core.pipeline_parallel.fb_overlap.modules.token_dispatcher import PackProb, UnpackProb
        MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.PackProb', PackProb)
        MindSporeAdaptation.register('mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.UnpackProb', UnpackProb)

        if args.gemm_gradient_accumulation_fusion:
            from mindspeed.mindspore.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
            MindSporeAdaptation.register('mindspeed.ops.npu_groupmatmul_add.npu_groupmatmul_add_fp32', npu_groupmatmul_add_fp32)
        from mindspeed.mindspore.ops.npu_matmul_add import npu_matmul_add_fp32
        MindSporeAdaptation.register('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32)
        MindSporeAdaptation.register('mindspeed.ops.npu_matmul_add.npu_matmul_add_fp32', npu_matmul_add_fp32)

        if args.use_moba_attn:
            from mindspeed_llm.mindspore.core.transformer.dot_product_attention import flash_attention_forward
            MindSporeAdaptation.register('mindspeed_llm.core.transformer.dot_product_attention.flash_attention_forward', flash_attention_forward)

        from mindspeed.mindspore.core.pipeline_parallel.schedules import deallocate_output_tensor_
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                                     deallocate_output_tensor_)

        if args.reuse_fp32_param:
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
            from mindspeed.mindspore.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
            target_func = DistributedOptimizer.__init__
            target_func_name = 'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__'
            clear_wrapper(target_func_name, target_func)
            MindSporeAdaptation.register(target_func_name, reuse_fp32_param_distrib_optimizer_init_wrapper)


    @staticmethod
    def reparse_args():
        """
        MindSpore extra arguments parser
        """
        from mindspeed_llm.mindspore.training.arguments import process_args_decorator
        MindSporeAdaptation.register('mindspeed_llm.training.arguments.process_args', process_args_decorator)
        MegatronAdaptation.apply()
        args = MegatronAdaptation.get_args()
        return args
