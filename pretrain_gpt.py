# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
<<<<<<< HEAD
import math
=======
from torch import nn
import torch.nn.functional as F
from torch import distributed as ptdist
from dist import setup_torch

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
<<<<<<< HEAD
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess

from torch import nn
import torch.nn.functional as F
=======
from deepspeed.accelerator import get_accelerator

import os
import time
import math
import subprocess
import logging
import wandb
from pathlib import Path
import socket

log = logging.getLogger(__name__)
HERE = Path(os.path.abspath(__file__)).parent


setup_torch(
    backend='deepspeed',
    port='5432',
)
log.info(f'Hello from {ptdist.get_rank()} / {ptdist.get_world_size()}')


def get_rank() -> int:
    return ptdist.get_rank()


def is_first_rank():
    return get_rank() == 0


WBRUN = None
if is_first_rank():
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR', None)
    if tensorboard_dir is not None:
        log.info(f'Patching tensorboard from {tensorboard_dir}')
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    
    # wbrun_id = wandb.util.generate_id()
    current_time = time.time()
    local_time = time.localtime(current_time)
    # os.environ['WANDB_RUN_GROUP'] = f'experiment-{generate_id()}'
    seq_len = os.environ.get('SEQ_LEN', 1)
    seq_len = int(seq_len) // 1024
    global_batch = os.environ.get('GLOBAL_BATCH', 1)
    mp_size = os.environ.get('MPSIZE', 1)
    pp_size = os.environ.get('PPSIZE', 1)
    world_size = ptdist.get_world_size()
    WBRUN = wandb.init(
        project='Megatron-DS',
        sync_tensorboard=True,
        dir=tensorboard_dir,
        resume='allow',
        # dir=os.getcwd(),
        # sync_tensorboard=True,
        # group=f'experiment-{generate_id()}'
        # group='long-seq-ANL0',
        name=f'{seq_len}kseq-len-no-seq-parallel-{global_batch}global-batch-{world_size}GPUs-{mp_size}MP-{pp_size}PP-{local_time.tm_hour}-{local_time.tm_min}'
        # name=f'{seq_len}kseq-len-new-{global_batch}global-batch-{world_size}GPUs-{mp_size}MP-{pp_size}PP-{local_time.tm_hour}-{local_time.tm_min}'
    )
    assert WBRUN is not None and WBRUN is wandb.run
    wandb.run.log_code(HERE.as_posix())  # type:ignore
    # WBRUN.log_code(HERE.as_posix())
    model_size = os.environ.get('MODEL_SIZE', None)
    if model_size is not None:
        WBRUN.config.update({'MODEL_SIZE': model_size})
    if WBRUN is not None:
        assert WBRUN is wandb.run
        WBRUN.config.update({'world_size': ptdist.get_world_size()})
        env = dict(os.environ)
        _ = env.pop('LS_COLORS', None)
        WBRUN.config.update({'env': env})
        hostname = socket.gethostbyaddr(socket.gethostname())[0]
        if hostname.startswith('theta'):
            WBRUN.config.update({'machine': 'ThetaGPU'})
        elif hostname.startswith('x3'):
            WBRUN.config.update({'machine': 'Polaris'})
        elif hostname.startswith('x1'):
            WBRUN.config.update({'machine': 'Sunspot'})
        else:
            WBRUN.config.update({'machine': hostname})

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)
<<<<<<< HEAD

    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
=======
    args = get_args()
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.config.update(vars(args))

    with deepspeed.zero.Init(
            data_parallel_group=mpu.get_data_parallel_group(),
            remote_device=None if args.remote_device == 'none' else args.remote_device,
            config_dict_or_path=args.deepspeed_config,
            enabled=args.zero_stage == 3,
            mpu=mpu
    ):
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
<<<<<<< HEAD

=======
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
<<<<<<< HEAD
=======
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.watch(
            model,
            log='all',
            log_graph=True,
        )

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

<<<<<<< HEAD
=======

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                    data['text'][:, -(current_seqlen+1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen+1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data

<<<<<<< HEAD
=======

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
<<<<<<< HEAD
    data_b = mpu.broadcast_data(keys, data, datatype)
=======
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
<<<<<<< HEAD
=======
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.log({'averaged_loss/averaged_loss': averaged_loss[0]})

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
    if args.mos or args.kd:
        # assert max(args.num_experts) >= 1
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
        elif args.kd:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'kd loss': mos_loss}
        print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}
<<<<<<< HEAD
=======

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677

def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
<<<<<<< HEAD
    
=======
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
    if teacher_model:
        with torch.no_grad():
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                # No need to truncate labels as we do not need it for the teacher logits
            tea_output, *tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            assert stu_output.size() == tea_output.size(), 'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(stu_output.size(), tea_output.size(), args.curriculum_seqlen)
<<<<<<< HEAD
=======

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    t0 = time.time()
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.log({'timers/batch-generator': time.time() - t0})

    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
            args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
            args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)
<<<<<<< HEAD

    if args.mos or args.kd:
        # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
        stu_output, *other_losses = model(tokens, position_ids, attention_mask)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            assert args.curriculum_seqlen is not None
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        output_tensor = mpu.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
    else:
        output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                            labels=labels)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

=======

    if args.mos or args.kd:
        # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
        stu_output, *other_losses = model(tokens, position_ids, attention_mask)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            assert args.curriculum_seqlen is not None
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        output_tensor = mpu.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
    else:
        output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                            labels=labels)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
    mos_loss = 0
    if args.mos or args.kd:
        assert model.training
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(args, stu_output,
                args.teacher_model[0], tokens, position_ids, attention_mask)
    
<<<<<<< HEAD
=======
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.log({'timers/forward_step': time.time() - t0})
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0
<<<<<<< HEAD


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()
=======


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    if is_first_rank():
        ds_report()
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


<<<<<<< HEAD
if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             data_post_process=data_post_process)
=======
def main():
    import deepspeed.comm as dist
    git_ds_info()
    t0 = time.time()
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        data_post_process=data_post_process,
        wbrun=WBRUN
    )
    dist.log_summary()
    if is_first_rank() and WBRUN is not None and WBRUN is wandb.run:
        WBRUN.log({'pretrain_time/pretrain_time': time.time() - t0})
        WBRUN.finish()


if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             data_post_process=data_post_process)
>>>>>>> 628e32bf8dc1d203bd4a5c1eaab92a25c8ec0677
