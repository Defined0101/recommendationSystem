import numpy as np
import os
import zipfile

# Klasör oluştur
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Kaynak npz dosyasını aç
input_file = 'recipe_data_gpu0.npz'

chunk_size = 500

with zipfile.ZipFile(input_file, 'r') as zf:
    with zf.open('ids.npy') as ids_file, zf.open('embeddings.npy') as embeddings_file:
        ids = np.load(ids_file)
        embeddings = np.load(embeddings_file)

        num_chunks = len(ids) // chunk_size + (1 if len(ids) % chunk_size else 0)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(ids))

            ids_chunk = ids[start_idx:end_idx]
            embeddings_chunk = embeddings[start_idx:end_idx]

            output_path = os.path.join(data_dir, f'chunk_{i}.npz')
            np.savez(output_path, ids=ids_chunk, embeddings=embeddings_chunk)

            print(f'{output_path} dosyası yazıldı.')
