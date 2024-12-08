import torch
import pandas as pd
import esm
import numpy as np

class FastaESM:
    def __init__(self,esm_model ):
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # disables dropout for deterministic results


    def preprocess_tokens(self, batch_tokens):
        max_length = 1000
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

    def encode_sequence(self, pdb_id, sequence):
        data = [(pdb_id, sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        batch_tokens = self.preprocess_tokens(batch_tokens)



        return batch_tokens

def main(csv_file_path, output_dir):
    fasta_esm = FastaESM(esm_model='esm2_t6_8M_UR50D')

    # Read data from CSV
    data = pd.read_csv(csv_file_path)

    # Iterate through each row in the CSV file
    for index, row in data.iterrows():
        pdb_id = row['pdbid']
        sequence = row['seq']

        # Encode sequence using ESM model
        batch_tokens = fasta_esm.encode_sequence(pdb_id, sequence).squeeze()

        # Save representation as .npy file
        output_file_path = f"{output_dir}/{pdb_id}.npy"
        np.save(output_file_path, batch_tokens)

        print(f"Saved representation for PDB ID {pdb_id} to {output_file_path}")

if __name__ == "__main__":
    csv_file_path = "training_seq.csv"  # Replace with the path to your CSV file
    output_dir = "token embedding-1000-8M"  # Specify the directory where you want to save the .npy files
    main(csv_file_path, output_dir)
