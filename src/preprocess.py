
import pandas as pd
import os
from eq_tokenizer import EquationTokenizer
import traceback
import logging

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/preprocess.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

def main():
    target_csv = "../FeynmanEquations.csv"
    if not os.path.exists(target_csv):
        target_csv = "FeynmanEquations.csv"
    msg = f"Loading {target_csv}..."
    print(msg)
    logging.info(msg)
    df = pd.read_csv(target_csv)
    df = df.dropna(subset=['Formula'])
    tokenizer = EquationTokenizer()
    msg = f"Found {len(df)} equations."
    print(msg)
    logging.info(msg)
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
                msg = f"\n[{row['Filename']}] {formula}\n  -> Tokens: {' '.join(tokens)}"
                print(msg)
                logging.info(msg)
        except Exception as e:
            msg = f"Failed to parse {formula}: {e}"
            print(msg)
            logging.error(msg)
            logging.error(traceback.format_exc())
            traceback.print_exc()
    out_df = pd.DataFrame(tokenized_data)
    out_path = "../tokenized_equations.csv"
    out_df.to_csv(out_path, index=False)
    msg = f"\nSaved tokenized data: {out_path}."
    print(msg)
    logging.info(msg)

if __name__ == "__main__":
    main()
