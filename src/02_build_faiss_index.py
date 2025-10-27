"""
02_build_faiss_index.py
------------------------
Builds a FAISS index from saved embeddings for fast similarity search.
"""

import os
import numpy as np
import faiss

# ----------------------------
# 1️⃣ Load Embeddings
# ----------------------------
embed_dir = os.path.join("..", "data", "embeddings")
embeddings_path = os.path.join(embed_dir, "embeddings.npy")

if not os.path.exists(embeddings_path):
    raise FileNotFoundError("❌ Embeddings not found! Run 01_extract_embeddings.py first.")

embeddings = np.load(embeddings_path).astype("float32")
print(f"✅ Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")

# ----------------------------
# 2️⃣ Build FAISS Index
# ----------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)

# ----------------------------
# 3️⃣ Save Index
# ----------------------------
index_path = os.path.join(embed_dir, "faiss_index.bin")
faiss.write_index(index, index_path)

print(f"✅ FAISS index saved to {index_path}")
