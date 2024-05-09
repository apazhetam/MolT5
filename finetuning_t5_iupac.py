import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field

from datasets import load_dataset
from tqdm import tqdm
    
def create_batches(dset, batch_size=32):
    num_rows = len(dset['train'])
    indices = np.random.permutation(num_rows)  # shuffleeee indices
    return [indices[int(i):int(i) + batch_size] for i in range(0, num_rows, batch_size)]

if __name__ == "__main__":
    dataset_inps = load_dataset("text", data_files={"train": ["/your/path/here"]}) # assume data is already preprocessed so that it has the questions and what not
    dataset_labels = load_dataset("text", data_files={"train": ["/your/path/here"]})
    
    num_epochs = 4
    batch_size = 32
    device = 'cuda'
    tokenizer = T5Tokenizer.from_pretrained("/your/path/here")
    model = T5ForConditionalGeneration.from_pretrained("/your/path/here").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    losses = []
    s = time.time()

    for epoch in range(num_epochs):
        batches = create_batches(dataset_inps, batch_size)  # Create shuffled batches at the beginning of each epoch
        num_steps_per_epoch = len(batches)
        
        for step, batch_indices in enumerate(batches):
            inps = [dataset_inps['train'][int(i)]['text'] for i in batch_indices]
            labels = [dataset_labels['train'][int(i)]['text'] for i in batch_indices]
            
            if step % 1000 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Step {step}/{num_steps_per_epoch}. It has been {time.time() - s} seconds since the last print.")
                s = time.time()

            input_ids = tokenizer(inps, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if step % 100000 == 0:
                model.save_pretrained(f"/scratch/gpfs/jomeike/thesis/models/t5_iupac_finetuning_epoch_{epoch}_step_{step}")

                with open('losses.txt', 'w') as f:
                    for loss in losses:
                        f.write(f"{loss}\n")
