# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
import os
import sys
from time import time
from functools import wraps
from logging import getLogger
import torch

from megatron.training import get_args
from megatron.core import mpu, dist_checkpointing
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper, FullyParallelLoadStrategyWrapper
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.training.utils import print_rank_0, unwrap_model, append_to_progress_log, is_last_rank
from megatron.training.async_utils import schedule_async_save
from megatron.training.checkpointing import (_load_base_checkpoint, get_rng_state, get_checkpoint_name,
                                             get_distributed_optimizer_checkpoint_name,
                                             ensure_directory_exists, generate_state_dict, get_checkpoint_tracker_filename)
from megatron.training.one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success
from megatron.training.checkpointing import read_metadata
from megatron.training.checkpointing import find_checkpoint_rank_0

from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_lora, merge_dicts, modify_keys_with_dict, filter_lora_keys
from mindspeed_llm.tasks.posttrain.utils import load_checkpoint_loosely

try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False


logger = getLogger(__name__)


def _load_base_checkpoint(load_dir, rank0=False, sharded_state_dict=None,
                          exit_on_missing_checkpoint=False, checkpoint_step=None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')

        # Conditionally exit if checkpoint not found.
        if exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    if checkpoint_step is not None:
        iteration = checkpoint_step
        release = False
    else:
        iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        is_dist_ckpt = checkpoint_name is not None and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                              return_base_dir=True)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        if not is_dist_ckpt:
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release,
                                                  return_base_dir=False)
        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    if is_dist_ckpt:
        if rank0:
            state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
            return state_dict, checkpoint_name, release

        # at this point args are available
        args = get_args()
        if sharded_state_dict is None:
            assert not args.auto_detect_ckpt_format and not args.use_dist_ckpt, (args.auto_detect_ckpt_format, args.use_dist_ckpt)
            raise RuntimeError('Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set.')

        load_strategy = get_default_load_sharded_strategy(checkpoint_name)
        if args.ckpt_fully_parallel_load:
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                             mpu.get_data_parallel_group(with_context_parallel=True))
        state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy, strict=args.dist_ckpt_strictness)
        return state_dict, checkpoint_name, release

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
    except ModuleNotFoundError:
        from megatron.legacy.fp16_deprecated import loss_scaler
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
        state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
        sys.modules.pop('megatron.model', None)
    except BaseException as e:
        print('could not load the checkpoint')
        print(e)
        sys.exit()

    return state_dict, checkpoint_name, release


def _load_base_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if getattr(args_, 'is_load_refer', False):
            kwargs['checkpoint_step'] = args_.refer_model_iter
        state_dict, checkpoint_name, release = fn(*args, **kwargs)
        rank0 = kwargs.pop('rank0')
        if is_enable_lora() and state_dict is not None:
            words_to_match = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
            exclude_words = ['base_layer', 'lora_', 'norm']
            state_dict = modify_keys_with_dict(state_dict, words_to_match, exclude_words)

            if not args_.lora_load or getattr(args_, 'is_load_refer', False):
                return state_dict, checkpoint_name, release

            # Read the tracker file and set the iteration.
            state_dict_lora, checkpoint_name_lora, release_lora = fn(args_.lora_load, rank0)
            if state_dict_lora is not None:
                merge_dicts(state_dict, state_dict_lora)
                checkpoint_name = checkpoint_name_lora
                release = release_lora
        return state_dict, checkpoint_name, release
    return wrapper


def load_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_enable_lora() or load_checkpoint_loosely():
            kwargs['strict'] = False
        return fn(*args, **kwargs)

    return wrapper


def load_args_from_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not isinstance(res, tuple):
            return res
        args, checkpoint_args = res
        
        def _set_arg(arg_name, old_arg_name=None, force=False):
            if not force and getattr(args, arg_name, None) is not None:
                return
            if old_arg_name is not None:
                checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
            else:
                checkpoint_value = getattr(checkpoint_args, arg_name, None)
            if checkpoint_value is not None:
                print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
                setattr(args, arg_name, checkpoint_value)
            else:
                print_rank_0(f"Checkpoint did not provide arguments {arg_name}")
        
        _set_arg('num_layer_list', force=True)
        _set_arg('post_norm', force=True)
        _set_arg('num_experts')
        _set_arg('sequence_parallel', force=True)
        _set_arg('n_shared_experts', force=True)
        _set_arg('qk_layernorm', force=True)
        _set_arg('moe_intermediate_size', force=True)
        _set_arg('first_k_dense_replace', force=True)
        _set_arg('moe_layer_freq', force=True)
        _set_arg('multi_head_latent_attention', force=True)
        _set_arg('qk_rope_head_dim', force=True)
        _set_arg('qk_nope_head_dim', force=True)
        _set_arg('q_lora_rank', force=True)
        _set_arg('kv_lora_rank', force=True)
        _set_arg('v_head_dim', force=True)
        _set_arg('shared_expert_gate', force=True)

        state_dict, checkpoint_name, release = _load_base_checkpoint(
            getattr(args, kwargs.get('load_arg', 'load')),
            rank0=True,
            exit_on_missing_checkpoint=kwargs.get('exit_on_missing_checkpoint', False),
            checkpoint_step=args.ckpt_step
        )
        checkpoint_version = state_dict.get('checkpoint_version', 0)
        if checkpoint_version >= 3.0:
            _set_arg('expert_model_parallel_size', force=True)
            
        return args, checkpoint_args
    
    return wrapper


def save_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, checkpointing_context=None,
                    pipeline_rank=None, expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None):
        """Save a model checkpoint.

        Checkpointing context is used to persist some checkpointing state
        throughout a single job. Must be initialized externally (not used if None).
        """
        start_ckpt = time()
        args = get_args()

        # Prepare E2E metrics at start of save checkpoint
        productive_metrics = on_save_checkpoint_start(args.async_save)

        # Only rank zero of the data parallel writes to the disk.
        model = unwrap_model(model)

        ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
        print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
            iteration, args.save, ckpt_format))

        # Collect rng state across data parallel ranks.
        rng_state = get_rng_state(args.use_dist_ckpt)

        # Checkpoint name.
        checkpoint_name = get_checkpoint_name(args.save, iteration, release=False, pipeline_parallel=pipeline_parallel,
                                              tensor_rank=tensor_rank, pipeline_rank=pipeline_rank,
                                              expert_parallel=expert_parallel, expert_rank=expert_rank,
                                              return_base_dir=args.use_dist_ckpt)

        # Save distributed optimizer's custom parameter state.
        if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
            optim_checkpoint_name = \
                get_distributed_optimizer_checkpoint_name(checkpoint_name)
            ensure_directory_exists(optim_checkpoint_name)
            optimizer.save_parameter_state(optim_checkpoint_name)

        async_save_request = None
        if args.async_save:
            if not args.use_dist_ckpt:
                raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
            elif args.dist_ckpt_format != 'torch_dist':
                raise NotImplementedError(
                    f'Async checkpoint save not implemented for {args.dist_ckpt_format} distributed checkpoint format')

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Collect args, model, RNG.
        if not torch.distributed.is_initialized() \
                or mpu.get_data_modulo_expert_parallel_rank(with_context_parallel=True) == 0 \
                or args.use_dist_ckpt:

            optim_sd_kwargs = {}
            if args.use_dist_ckpt and args.use_distributed_optimizer:
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                    if args.ckpt_fully_parallel_save
                                                    else 'dp_zero_gather_scatter')
                print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
            state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                             args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

            state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
            if args.use_dist_ckpt:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    ensure_directory_exists(checkpoint_name, check_parent=False)
                validate_sharding_integrity = True
                save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                                  get_default_save_sharded_strategy(
                                                                      args.dist_ckpt_format))
                if args.ckpt_assume_constant_structure and args.dist_ckpt_format == 'torch_dist':
                    save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
                if args.ckpt_fully_parallel_save:
                    if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                        # Already saved once before - don't need to rerun sharding validation
                        validate_sharding_integrity = not args.ckpt_assume_constant_structure
                    else:
                        save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(
                            with_context_parallel=True),
                                                                         args.ckpt_assume_constant_structure)
                # Store save strategy for future checkpoint saves
                if checkpointing_context is not None:
                    checkpointing_context['save_strategy'] = save_strategy
                end_ckpt = time()
                logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
                async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                             async_sharded_save=args.async_save)

                # [ModelOpt]: save sharded modelopt_state
                if has_nvidia_modelopt:
                    save_sharded_modelopt_state(model, checkpoint_name, (args.dist_ckpt_format, 1))
            else:
                # [ModelOpt]: Inject modelopt_state into state_dict
                if has_nvidia_modelopt:
                    save_modelopt_state(model, state_dict)
                # If only save lora ckpt
                if args.lora_ckpt_filter:
                    state_dict = filter_lora_keys(state_dict)
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)
        start_misc = time()
        if not args.async_save:
            if async_save_request is not None:
                raise ValueError("async_save_request should be None")
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # And update the latest iteration
        if not torch.distributed.is_initialized() \
                or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)

            def iter_finalize_fn():
                with open(tracker_filename, 'w') as f:
                    f.write(str(iteration))
                print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                             .format(iteration, args.save))
                if args.log_progress and args.async_save:
                    append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                           barrier=False)

            if args.async_save:
                if async_save_request is None:
                    raise ValueError("async_save_request should be None")
                async_save_request.add_finalize_fn(iter_finalize_fn)
            else:
                iter_finalize_fn()

        # Additional callback for one_logger (last rank)
        if not torch.distributed.is_initialized() \
                or is_last_rank():
            def onelogger_finalize_fn():
                on_save_checkpoint_success(productive_metrics, args.async_save)

            if args.async_save:
                if async_save_request is None:
                    raise ValueError("async_save_request should be None")
                async_save_request.add_finalize_fn(onelogger_finalize_fn)
            else:
                onelogger_finalize_fn()

        if args.async_save:
            schedule_async_save(async_save_request)
            print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                         .format(iteration, args.save))

        # Wait so everyone is done (not necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        end_misc = time()
        logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")
    return wrapper