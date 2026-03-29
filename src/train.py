import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import FeynmanDataset, mask_tokens_for_jepa
from model import MathTransformer
import torch.optim as optim

def train_jepa(model, dataloader, optimizer, epochs=5, device='cpu'):
    model.train()
    criterion = nn.SmoothL1Loss()
    
    print("--- Starting JEPA Pretraining ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (X, y, tokens) in enumerate(dataloader):
            tokens = tokens.to(device)
            context, targets = mask_tokens_for_jepa(tokens, mask_token_id=1, pad_token_id=0)
            
            optimizer.zero_grad()
            pred, tgt = model.forward_jepa(context, targets)
            
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            
            model.update_teacher()
            
            epoch_loss += loss.item()
            
        print(f"JEPA Phase Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(dataloader):.4f}")

def train_autoregressive(model, dataloader, optimizer, epochs=10, device='cpu'):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("--- Starting Symbolic Regression Finetuning Phase ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (X, y, tokens) in enumerate(dataloader):
            X, y, tokens = X.to(device), y.to(device), tokens.to(device)
            
            tgt_in = tokens[:, :-1]
            tgt_out = tokens[:, 1:]
            
            optimizer.zero_grad()
            logits = model.forward_autoregressive(X, y, tgt_in)
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Causal Output Phase Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Framework on {device}...")
    
    dataset = FeynmanDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MathTransformer(vocab_size=len(dataset.vocab), max_seq_len=64, max_dims=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)    
    
    train_jepa(model, dataloader, optimizer, epochs=2, device=device)
    
    train_autoregressive(model, dataloader, optimizer, epochs=5, device=device)
    
    torch.save(model.state_dict(), "../jepa_symreg_model.pt")
    print("Model saved.")
