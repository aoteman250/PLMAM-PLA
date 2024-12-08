import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer,BertTokenizerFast, BertModel

model = AutoModel.from_pretrained(".../MoLFormer-XL-both-10pct",trust_remote_code=True)#Replace with the path to your file
tokenizer = AutoTokenizer.from_pretrained(".../MoLFormer-XL-both-10pct",trust_remote_code=True)

class FastaESM:
    def __init__(self):
        pass

    def preprocess_tokens(self, batch_tokens):
        max_length = 120
        batch_size, seq_length = batch_tokens.shape
        if seq_length < max_length:
            # Padding with zeros
            padding_length = max_length - seq_length
            padding = torch.zeros((batch_size, padding_length), dtype=torch.long)
            batch_tokens = torch.cat([batch_tokens, padding], dim=1)
        elif seq_length > max_length:
            # Truncating
            batch_tokens = batch_tokens[:, :max_length]
        return batch_tokens

    def encode_sequence(self, smiles):
        inputs = tokenizer(smiles, return_tensors="pt")
        inputs = inputs['input_ids']
        inputs = self.preprocess_tokens(inputs)
        # inputs = model(inputs)
        return inputs

def main(csv_file_path, output_dir):
    fasta_esm = FastaESM()

    # Read data from CSV
    data = pd.read_csv(csv_file_path)

    # Iterate through each row in the CSV file
    for index, row in data.iterrows():
        pdb_id = row['pdbid']
        smiles = row['smiles']

        # Encode sequence using ESM model
        batch_tokens = fasta_esm.encode_sequence(smiles).squeeze()

        # Save representation as .npy file
        output_file_path = f"{output_dir}/{pdb_id}.npy"
        np.save(output_file_path, batch_tokens.numpy())

        print(f"Saved representation for PDB ID {pdb_id} to {output_file_path}")

if __name__ == "__main__":
    csv_file_path = "training_smi.csv"  # Replace with the path to your CSV file
    output_dir = "Molformer"  # Specify the directory where you want to save the .npy files
    main(csv_file_path, output_dir)
