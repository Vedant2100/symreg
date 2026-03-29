
import torch
import numpy as np
import sympy as sp
from dataloader import FeynmanDataset
import logging
import os

# Setup logging
os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    filename='../logs/evaluate.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

def stringify_tokens(token_ids, inv_vocab):
    words = []
    for tid in token_ids:
        tid = tid.item()
        if tid in [0, 2]: continue
        if tid == 3: break
        words.append(inv_vocab[tid])
    return words

def prefix_to_sympy(tokens):
    if not tokens: 
        return None
    t = tokens.pop(0)
    
    if t == 'pi': return sp.pi
    if t == 'E': return sp.E
    
    if t == 'add':
        left = prefix_to_sympy(tokens)
        right = prefix_to_sympy(tokens)
        if left is None or right is None: return None
        return left + right
        
    if t == 'mul':
        left = prefix_to_sympy(tokens)
        right = prefix_to_sympy(tokens)
        if left is None or right is None: return None
        return left * right
        
    if t == 'pow':
        base = prefix_to_sympy(tokens)
        power = prefix_to_sympy(tokens)
        if base is None or power is None: return None
        return base ** power
        
    unary_funcs = {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
        'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'ln': sp.log
    }
    
    if t in unary_funcs:
        arg = prefix_to_sympy(tokens)
        if arg is None: return None
        return unary_funcs[t](arg)
        
    try:
        val = sp.Rational(t) if '/' in t else sp.Float(t) if '.' in t else sp.Integer(t)
        return val
    except:
        pass
        
    if t == '<C>':
        return sp.Symbol('C')
        
    return sp.Symbol(t)

def verify_symbolic_equivalence(pred_tokens, true_formula_str):
    try:
        pred_expr = prefix_to_sympy(pred_tokens.copy())
        if pred_expr is None: return False
        
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = standard_transformations + (implicit_multiplication_application,)
        true_expr = parse_expr(true_formula_str.replace('^', '**'), transformations=transformations)
        
        diff = sp.simplify(pred_expr - true_expr)
        return diff == 0
    except Exception as e:
        return False
        
def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    exact_matches = 0
    symbolic_matches = 0
    total = 0
    
    print("Evaluating Model on Ground Truth Symbolic Equivalence...")
    with torch.no_grad():
        for batch_idx, (X, y, tokens) in enumerate(dataloader):
            X, y, tokens = X.to(device), y.to(device), tokens.to(device)
            B = X.shape[0]
            
            max_len = tokens.shape[1]
            preds = torch.full((B, 1), 2, dtype=torch.long, device=device) # 2 is <START>
            
            for step in range(max_len - 1): #greedy
                logits = model.forward_autoregressive(X, y, preds)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                preds = torch.cat([preds, next_tokens], dim=1)
            
            for i in range(B):
                pred_seq = stringify_tokens(preds[i], dataloader.dataset.inv_vocab)
                tgt_seq = stringify_tokens(tokens[i], dataloader.dataset.inv_vocab)
                
                if pred_seq == tgt_seq:
                    exact_matches += 1
                    symbolic_matches += 1
                else:
                    true_formula = dataloader.dataset.eq_df.iloc[batch_idx * dataloader.batch_size + i]['Original_Formula']
                    if verify_symbolic_equivalence(pred_seq, true_formula):
                        symbolic_matches += 1
                        
                if i == 0 and batch_idx == 0:
                    msg = f"Example Generation --->\nTarget: {' '.join(tgt_seq)}\nPreds : {' '.join(pred_seq)}\n"
                    print(msg)
                    logging.info(msg)
                        
                total += 1
                
    print("="*40)
    print(f"Total Equations Evaluated: {total}")
    print(f"Exact Sequence Match Accuracy: {exact_matches/total * 100:.2f}%")
    print(f"Symbolic Equivalence Accuracy: {symbolic_matches/total * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    from model import MathTransformer
    from torch.utils.data import DataLoader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FeynmanDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MathTransformer(vocab_size=len(dataset.vocab), max_seq_len=64, max_dims=10).to(device)
    
    try:
        model.load_state_dict(torch.load("../jepa_symreg_model.pt", map_location=device))
        print("Pretrained weights loaded successfully.")
    except:
        print("No saved model found, evaluating blank weights.")
        
    evaluate_model(model, dataloader, device=device)
