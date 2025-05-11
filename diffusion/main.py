# Standard library
import copy
import gc
import logging
import os
import sys
import warnings

# Ensure project paths are on PYTHONPATH
sys.path.append('/tahoe/diffusePerturb')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party libraries
import fsspec
import hydra
import lightning as L
import omegaconf
from omegaconf import DictConfig, OmegaConf as om
import rich.syntax
import rich.tree
from rich.traceback import install
import torch

from composer import utils as composer_utils
from composer.core.callback import Callback
from composer.utils import dist, get_device, reproducibility

from llmfoundry.registry import callbacks
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_logger,
    build_optimizer,
    build_scheduler,
)
from llmfoundry.utils.config_utils import (
    log_config,
    pop_config,
    process_init_device,
    update_batch_size_info,
)

from streaming.base.util import clean_stale_shared_memory

# MosaicFM local modules
import dataloader
import diffusion
import utils
from mosaicfm.data import build_dataloader
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import CellClassification, MarginalEssentiality
from mosaicfm.tokenizer import GeneVocab
from mosaicfm.utils import download_file_from_s3_url

# Register tasks & install traceback handler
callbacks.register("cell-classification", func=CellClassification)
callbacks.register("marginal-essentiality", func=MarginalEssentiality)
install()







omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
    logger.info('Starting Zero Shot Eval.')

    model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _train(config, train_dl, valid_dl, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    model = diffusion.Diffusion(
        config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_dl, valid_dl, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    train_loader_config = config.train_loader 
    valid_loader_config = config.valid_loader 


    prev_model_config = config.prev_model 
    vocab_config = config.vocabulary 
    collator_config = config.collator 
    device_train_batch_size: int = config.device_train_batch_size

    device_eval_batch_size: int = config.device_eval_batch_size


    # Build vocab
    vocab = GeneVocab.from_file('/tahoe/diffusePerturb/vocab.json')
    special_tokens = ["<pad>", "<cls>", "<eoc>"]

    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    if collator_config.get("use_junk_tokens", False):
        # Based on Karpathy's observation that 64 is a good number for performance
        # https://x.com/karpathy/status/1621578354024677377?s=20
        original_vocab_size = len(vocab)
        remainder = original_vocab_size % 64
        if remainder > 0:
            junk_tokens_needed = 64 - remainder
            for i in range(junk_tokens_needed):
                junk_token = f"<junk{i}>"
                vocab.append_token(junk_token)

    ## Update PAD token ID
    collator_config.pad_token_id = vocab["<pad>"]
    ## Update model config with Vocab Size
    prev_model_config.vocab_size = len(vocab)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    clean_stale_shared_memory()
    train_loader = build_dataloader(
        vocab=vocab,
        loader_cfg=train_loader_config,
        collator_cfg=collator_config,
        device_batch_size=device_train_batch_size,
    )

    valid_loader = build_dataloader(
        vocab=vocab,
        loader_cfg=valid_loader_config,
        collator_cfg=collator_config,
        device_batch_size=device_eval_batch_size,
    )
    

    '''
    if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
    elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
    else:
    '''
    _train(config, train_loader, valid_loader, logger, tokenizer)


if __name__ == '__main__':
  main()