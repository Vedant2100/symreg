import pandas as pd
import os
from eq_tokenizer import EquationTokenizer
import traceback

def main():
    target_csv = "../FeynmanEquations.csv"
    if not os.path.exists(target_csv):
        target_csv = "FeynmanEquations.csv"
        
    print(f"Loading {target_csv}...")
    df = pd.read_csv(target_csv)
    
    df = df.dropna(subset=['Formula'])
    
    tokenizer = EquationTokenizer()
    
    print(f"Found {len(df)} equations.")
    
    tokenized_data = []
    
    for idx, row in df.iterrows():
        formula = str(row['Formula']).strip()
        try:
            tokens = tokenizer.tokenize_formula(formula)
            tokenized_data.append({
                'Filename': row['Filename'],
                'Original_Formula': formula,
                'Tokens': " ".join(tokens)
            })
            if idx < 5:
                print(f"\n[{row['Filename']}] {formula}")
                print(f"  -> Tokens: {' '.join(tokens)}")
        except Exception as e:
            print(f"Failed to parse {formula}: {e}")
            traceback.print_exc()
            
    out_df = pd.DataFrame(tokenized_data)
    out_path = "../tokenized_equations.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved tokenized data: {out_path}.")

if __name__ == "__main__":
    main()
