import argparse
import os
import gc
import pandas as pd
import evaluate
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loguru import logger
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Iterator, Dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from modules.dataset.dataset import dataset
from modules.model.gpt2 import GPT2Model

from config.config import Config

class GPT2_TextSum(pl.LightningModule):
    def __init__(self, config: Config, random_state: int):
        super(GPT2_TextSum, self).__init__()
        self.config = config
        self.gpt2 = GPT2Model(config=self.config.model_args)        
        logger.info(f"Loading tokenizer: {self.config.model_args.model_checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_args.model_checkpoint)
        self.tokenizer.do_lower_case = True
        
        self.log_dir = self.config.trainer_args.log
        self.checkpoint = self.config.trainer_args.checkpoint

        self.loss_func = self.configure_loss_func(loss_func=self.config.trainer_args.losses)
        
        self.num_training_steps = len(self.train_dataloader()) // self.config.trainer_args.accumulate_grad_batches * self.config.trainer_args.max_epochs
        self.num_warmup_steps = int(self.config.trainer_args.warmup_ratio * self.num_training_steps)
        self.random_state = random_state

    def configure_loss_func(self, loss_func: str) -> nn.Module:
        """Create loss function based on ``loss_func`` name

        Args:
            loss_func (str): the name of loss function

        Returns:
            nn.Module: loss function
        """
        if loss_func not in ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']:
            raise ValueError(f"{loss_func} is not supported. Supported loss functions are: ['CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss']")

        if loss_func == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_func == "NLLLoss":
            return nn.NLLLoss()
        elif loss_func == "BCELoss":
            return nn.BCELoss()
        elif loss_func == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
    
    def make_dir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.system(f'mkdir -p {dir_path}')
            os.system(f"chmod -R 777 {dir_path}")
    
    def setup(self, stage):
        self.make_dir(dir_path=self.log_dir)
        self.make_dir(dir_path=self.checkpoint)
        
        self.freeze_layers()
        
        return True
    
    def get_dataloader(self, mode: str):

        return dataset(tokenizer=self.tokenizer,
                       model=mode,
                       file_path=self.config.dataset_args.file_path,
                       max_seq_length=self.config.dataset_args.max_seq_length,
                       random_state=self.random_state,
                       batch_size=self.config.trainer_args.batch_size,
                       num_workers=self.config.trainer_args.num_workers)

    def train_dataloader(self):
        return self.get_dataloader(mode='train')

    def val_dataloader(self):
        return self.get_dataloader(mode='valid')

    def test_dataloader(self):
        return self.get_dataloader(mode='test')
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                                       num_warmup_steps=self.num_warmup_steps,
                                                                                       num_training_steps=self.num_training_steps)
        
        return scheduler
    
    def configure_optimizers(self):
        param_optimizer = [[name, param] for name, param in self.gpt2.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.config.trainer_args.no_decay)],
             'weight_decay': self.config.trainer_args.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.config.trainer_args.no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer: torch.optim.Optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.trainer_args.lr)
        scheduler = self.configure_scheduler(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def compute_loss(self, batch: Iterator):
        logits = self.gpt2(batch['input_ids'], self.tokenizer.pad_token_id)
        logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))

        loss = self.loss_func(logits, batch['label'][:, 1:].contiguous().view(-1))

        return loss
        
    def training_step(self, batch: Iterator, batch_idx):
        loss: torch.Tensor = self.compute_loss(batch=batch)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: Iterator, batch_idx):
        loss = self.compute_loss(batch=batch)

        self.log('val_loss', loss)
        
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([batch for batch in validation_step_outputs]).mean()
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def forward(self, x: Dict[str, torch.Tensor]):
        outputs = self.exab.model.generate(
            input_ids=x['src_abs_input_ids'].to(self.device),
            max_length=self.dataset_args.tgt_max_length,
            attention_mask=x['src_abs_attention_mask'].to(self.device),
            num_beams=self.trainer_args.num_beams
        )
        outputs = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]
        # Replace -100 in the labels as we can't decode them
        labels = torch.where(x['decoder_input_ids'][:, 1:] != -100,
                             x['decoder_input_ids'][:, 1:], 
                             self.tokenizer.pad_token_id)
        
        actuals = [self.tokenizer.decode(lb, clean_up_tokenization_spaces=False, skip_special_tokens=True) for lb in labels]

        return outputs, actuals
        
    def predict_step(self, batch: Iterator, batch_idx):
        outputs, actuals = self.forward(x=batch)
        metrics = evaluate.load('rouge')
        results = metrics.compute(predictions=outputs, references=actuals)
        return results



# ----- MAIN process -----
def main(config: Config, project: str, random_state: int):
    wandb_logger = WandbLogger(project=project, save_dir=config.trainer_args.log)
    early_stopping = EarlyStopping(
        monitor=config.trainer_args.monitor,
        mode='min',
        min_delta=config.trainer_args.delta,
        patience=config.trainer_args.patience,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.trainer_args.checkpoint,
        filename=config.model_name+'-{epoch}-{step}-{val_loss:.2f}',
        monitor=config.trainer_args.monitor,
        mode='min',
        save_top_k=config.trainer_args.save_top_k,
        save_on_train_epoch_end=config.trainer_args.save_on_train_epoch_end
    )
    
    trainer = Trainer(
        enable_progress_bar=False,
        accelerator=config.trainer_args.accelerator,
        devices=config.trainer_args.devices,
        accumulate_grad_batches=config.trainer_args.accumulate_grad_batches,
        amp_backend=config.trainer_args.amp_backend,
        auto_lr_find=config.trainer_args.auto_lr_find,
        auto_scale_batch_size=config.trainer_args.auto_scale_batch_size,
        auto_select_gpus=config.trainer_args.auto_select_gpus,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=config.trainer_args.checkpoint,
        enable_model_summary=config.trainer_args.enable_model_summary,
        enable_checkpointing=config.trainer_args.enable_checkpointing,
        max_epochs=config.trainer_args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.trainer_args.eval_steps,
        precision=config.trainer_args.precision
    )
    
    gc.collect()
    model = GPT2_TextSum(config, random_state=random_state)
    trainer.fit(model)
    
    # logger.info('----- Testing -----')
    # predictions = trainer.predict(dataloaders=model.test_dataloader(), ckpt_path='best')
    # rouge_scores = pd.DataFrame(predictions).mean().to_dict()
    # logger.info(rouge_scores)


 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--project', type=str, default='gpt-2-text-sum')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_file', type=str, default='./config/config.ini', help='The configuration file.')
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    config = Config(config_file=args.config_file)
    main(config=config, project=args.project, random_state=args.random_state)
    