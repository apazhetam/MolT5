import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import time
from datasets import load_dataset

def batch_corrupt_texts(input_texts, replace_percentage=0.15, max_extra_id=99):
    corrupted_texts = []
    target_texts = []
    extra_id_counter = 0  

    for text in input_texts:
        tokens = tokenizer.tokenize(text)
        num_tokens = len(tokens)
        num_to_replace = max(1, int(num_tokens * replace_percentage))

        replace_indices = np.sort(np.random.choice(num_tokens, num_to_replace, replace=False))

        corrupted_text = []
        target_text = []

        i = 0
        while i < num_tokens:
            if i in replace_indices:
                start = i
                while i in replace_indices and i < num_tokens:
                    i += 1
                end = i
                if extra_id_counter <= max_extra_id:  
                    corrupted_text.append(f"<extra_id_{extra_id_counter}>")
                    target_text.append(f"<extra_id_{extra_id_counter}> " + ' '.join(tokens[start:end]).replace("▁", ""))
                    extra_id_counter += 1
            else:
                corrupted_text.append(tokens[i])
                i += 1

        corrupted_text_str = ' '.join(corrupted_text).replace("▁", " ").strip()
        target_text_str = ' '.join(target_text).replace("▁", " ").strip()

        corrupted_texts.append(corrupted_text_str)
        target_texts.append(target_text_str)

    return corrupted_texts, target_texts

def mini_batch(dataset, batch_size=1):
    indices = np.random.randint(0, len(dataset), size=batch_size)
    batch_texts = [dataset['train'][int(i)]['text'] for i in indices]
    return batch_texts

def get_corrupteds_and_targets(batch):
    corrupted_texts, target_texts = batch_corrupt_texts(batch)
    return corrupted_texts, target_texts
    
def create_batches(dataset, batch_size=32):
    num_rows = len(dataset['train'])
    indices = np.random.permutation(num_rows)  # shufffleee 
    return [indices[int(i):int(i) + batch_size] for i in range(0, num_rows, batch_size)]

if __name__ == "__main__":
    dataset = load_dataset("text", data_files={"train": ["/your/path/here"]})
    
    num_epochs = 2
    batch_size = 32
    device = 'cuda'
    tokenizer = T5Tokenizer.from_pretrained("/your/path/here")
    model = T5ForConditionalGeneration.from_pretrained("/your/path/here").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    losses = []
    s = time.time()

    for epoch in range(num_epochs):
        scheduler.step()  # update learning rate
        batches = create_batches(dataset, batch_size)  
        num_steps_per_epoch = len(batches)
        
        for step, batch_indices in enumerate(batches):
            batch_texts = [dataset['train'][int(i)]['text'] for i in batch_indices]
            
            if step % 1000 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Step {step}/{num_steps_per_epoch}. It has been {time.time() - s} seconds since the last print.")
                s = time.time()

            corrupted_texts, target_texts = get_corrupteds_and_targets(batch_texts)
            input_ids = tokenizer(corrupted_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if step % 100000 == 0:
                model.save_pretrained(f"./t5_iupac_pretraining_epoch_{epoch}_step_{step}")

                with open('losses.txt', 'w') as f:
                    for loss in losses:
                        f.write(f"{loss}\n")
