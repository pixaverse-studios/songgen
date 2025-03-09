import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
)
import logging
from typing import Dict, List, Optional, Tuple, Union
import wandb
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from songgen.models.mixed import SongGenMixedForConditionalGeneration
from songgen.models.configuration import SongGenConfig, SongGenDecoderConfig
from songgen.data.dataset import SongGenDataset, SongGenDataCollator
from songgen.tokenizers.lyrics.lyrics_tokenizer import VoiceBpeTokenizer

@dataclass
class SongGenTrainingArguments(TrainingArguments):
    max_audio_length: int = field(default=480000, metadata={"help": "Maximum audio length in samples"})
    max_text_length: int = field(default=256, metadata={"help": "Maximum text length in tokens"})
    max_lyrics_length: int = field(default=512, metadata={"help": "Maximum lyrics length in tokens"})
    data_dir: str = field(default=None, metadata={"help": "Path to data directory"})
    model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co"})
    description_tokenizer_name_or_path: str = field(default="t5-small", metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW"})
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps"})
    num_train_epochs: int = field(default=10, metadata={"help": "Total number of training epochs to perform"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps"})
    eval_steps: int = field(default=1000, metadata={"help": "Run evaluation every X steps"})
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps"})
    save_total_limit: int = field(default=5, metadata={"help": "Limit the total amount of checkpoints"})
    fp16: bool = field(default=True, metadata={"help": "Whether to use 16-bit (mixed) precision training"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing to save memory"})

class SongGenTrainer:
    def __init__(
        self,
        model: SongGenMixedForConditionalGeneration,
        args: SongGenTrainingArguments,
        train_dataset: SongGenDataset,
        eval_dataset: Optional[SongGenDataset] = None,
        data_collator: Optional[SongGenDataCollator] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.training_start_time = None
        self.samples_processed = 0

        # Initialize distributed training if needed
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            self.model = self.model.cuda(args.local_rank)
            
            # Enable gradient checkpointing before DDP wrapping if needed
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            self.model = DDP(self.model, 
                           device_ids=[args.local_rank],
                           find_unused_parameters=True)
            
            # Set static graph mode since our model architecture doesn't change during training
            # This is needed for distributed training
            self.model._set_static_graph()
            
            self.train_sampler = DistributedSampler(train_dataset)
        else:
            self.model = self.model.cuda()
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            self.train_sampler = None

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataset) // (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        
        # Account for distributed training
        if args.local_rank != -1:
            num_update_steps_per_epoch = num_update_steps_per_epoch // torch.distributed.get_world_size()
            
        self.total_training_steps = num_update_steps_per_epoch * args.num_train_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.total_training_steps,
        )

        # Initialize wandb if main process
        if args.local_rank in [-1, 0]:
            wandb.init(project="songgen-training", config=args)

    def train(self):
        self.model.train()
        self.training_start_time = torch.cuda.Event(enable_timing=True)
        self.training_end_time = torch.cuda.Event(enable_timing=True)
        self.training_start_time.record()
        
        # Log effective batch size
        global_batch_size = self.args.per_device_train_batch_size
        if self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
            global_batch_size *= world_size
        if self.args.gradient_accumulation_steps > 1:
            global_batch_size *= self.args.gradient_accumulation_steps
            
        if self.args.local_rank in [-1, 0]:
            print(f"\nTraining with:")
            print(f"  Per-GPU batch size: {self.args.per_device_train_batch_size}")
            print(f"  Total batch size: {global_batch_size}")
            print(f"  Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            print(f"  Total optimization steps: {self.total_training_steps}\n")
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )

        progress_bar = tqdm(
            total=self.total_training_steps,
            disable=self.args.local_rank not in [-1, 0],
        )
        
        completed_steps = 0
        for epoch in range(self.args.num_train_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
                
            for step, batch in enumerate(train_dataloader):
                # Move batch to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                if outputs.vocal_loss is not None:
                    loss = loss + outputs.vocal_loss  # Add vocal loss if available
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # Backward pass
                if self.args.fp16:
                    with torch.cuda.amp.autocast():
                        loss.backward()
                else:
                    loss.backward()

                # Update weights if gradient accumulation is complete
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Log gradient norms for debugging
                    if completed_steps % self.args.logging_steps == 0 and self.args.local_rank in [-1, 0]:
                        total_norm = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        wandb.log({"gradient_norm": total_norm})
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1
                    progress_bar.update(1)

                # Logging
                if completed_steps % self.args.logging_steps == 0:
                    if self.args.local_rank in [-1, 0]:
                        # Calculate throughput
                        self.training_end_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = self.training_start_time.elapsed_time(self.training_end_time) / 1000  # Convert to seconds
                        
                        # Calculate samples per second
                        samples_per_step = global_batch_size
                        total_samples = completed_steps * samples_per_step
                        throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
                        
                        logs = {
                            "loss": loss.item() * self.args.gradient_accumulation_steps,
                            "lr": self.scheduler.get_last_lr()[0],
                            "step": completed_steps,
                            "samples_per_second": throughput,
                            "total_samples": total_samples,
                            "elapsed_time": elapsed_time,
                        }
                        if outputs.vocal_loss is not None:
                            logs["vocal_loss"] = outputs.vocal_loss.item()
                        if outputs.codebook_losses is not None:
                            for i, cb_loss in enumerate(outputs.codebook_losses):
                                logs[f"codebook_{i}_loss"] = cb_loss.item()
                        
                        # Print performance metrics
                        print(f"\nStep {completed_steps}:")
                        print(f"  Samples/second: {throughput:.2f}")
                        print(f"  Total samples: {total_samples}")
                        print(f"  Elapsed time: {elapsed_time:.2f}s")
                        print(f"  Loss: {logs['loss']:.4f}")
                        
                        wandb.log(logs)

                # Evaluation
                if self.eval_dataset is not None and completed_steps % self.args.eval_steps == 0:
                    self.evaluate()

                # Saving
                if completed_steps % self.args.save_steps == 0:
                    if self.args.local_rank in [-1, 0]:
                        self.save_model(completed_steps)

                if completed_steps >= self.total_training_steps:
                    break

            if completed_steps >= self.total_training_steps:
                break

        # Final save
        if self.args.local_rank in [-1, 0]:
            self.save_model(completed_steps, final=True)

    def evaluate(self):
        self.model.eval()
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )

        total_eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                total_eval_loss += outputs.loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        #if self.args.local_rank in [-1, 0]:
            #wandb.log({"eval_loss": avg_eval_loss})
        print(f"Eval loss: {avg_eval_loss}")

        self.model.train()

    def save_model(self, step, final=False):
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        save_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        if final:
            save_dir = os.path.join(self.args.output_dir, "final")

        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

def main():
    # Parse arguments
    parser = HfArgumentParser(SongGenTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Set random seed
    set_seed(args.seed)

    # When using torchrun, we only need to set the device
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        # Log distributed training info
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(f"[GPU {rank}] Initializing process group: world_size = {world_size}")
        
        # Log GPU info
        device = torch.cuda.current_device()
        print(f"[GPU {rank}] Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"[GPU {rank}] Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")

    # Load tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.description_tokenizer_name_or_path,
        padding_side='right'
    )
    lyrics_tokenizer = VoiceBpeTokenizer()

    # Load model and config
    text_encoder_config = {
        "model_type": "t5",  # Required: specify the text encoder type
        "vocab_size": text_tokenizer.vocab_size,  # Required: match tokenizer vocab size
    }

    # Create decoder config using the class
    decoder_config = SongGenDecoderConfig(
        vocab_size=1088,  # Required: 1024 (codec vocab size) + 64 
        max_position_embeddings=6000,  # Non-default: increased from default 2048
        track_pattern="mixed",  # Required: specify generation pattern
        use_cache=False,  # Disable caching when using gradient checkpointing
        gradient_checkpointing=True  # Enable gradient checkpointing
    )

    config = SongGenConfig(
        prompt_cross_attention=True,  # Non-default: enable cross-attention for prompts
        add_prenet=True,  # Non-default: enable prenet for lyrics
        text_encoder=text_encoder_config,  # Required
        decoder=decoder_config.to_dict(),  # Convert config to dict as required by SongGenConfig
        vocab_size=len(lyrics_tokenizer),  # Get vocab size using __len__ method
        decoder_start_token_id=1025,
        pad_token_id=1024,   # padding token can be the same as bos_token_id
        bos_token_id=1025,  # Beginning of sequence token
        eos_token_id=1024,  # End of sequence token
        is_encoder_decoder=True,  # This is an encoder-decoder model
        num_codebooks=8,  # Number of codebooks from XCodec
        use_cache=False,  # Disable caching when using gradient checkpointing
        gradient_checkpointing=True  # Enable gradient checkpointing
    )

    model = SongGenMixedForConditionalGeneration(config)

    # Enable gradient checkpointing at model level
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Disable model output caching when using gradient checkpointing
        model.config.use_cache = False

    # Create datasets
    train_dataset = SongGenDataset(
        data_dir=args.data_dir,
        split="train",
        text_tokenizer=text_tokenizer,
        max_text_length=args.max_text_length,
        max_lyrics_length=args.max_lyrics_length,
        max_audio_length=args.max_audio_length,
    )
    print(f"Train dataset: {len(train_dataset)}")
    
    eval_dataset = SongGenDataset(
        data_dir=args.data_dir,
        split="eval",
        text_tokenizer=text_tokenizer,
        max_text_length=args.max_text_length,
        max_lyrics_length=args.max_lyrics_length,
        max_audio_length=args.max_audio_length,
    )

    # Create data collator
    data_collator = SongGenDataCollator(
        text_tokenizer=text_tokenizer,
        lyrics_tokenizer=lyrics_tokenizer,
        max_text_length=args.max_text_length,
        max_lyrics_length=args.max_lyrics_length,
        max_audio_length=args.max_audio_length,
        pad_token_id=text_tokenizer.pad_token_id,
    )

    # Initialize trainer
    trainer = SongGenTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=text_tokenizer,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 