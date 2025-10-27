"""
03_search_similar.py
--------------------
Search for visually similar fashion images using FAISS.
"""

import os
import faiss
import torch
import numpy as np
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
embed_dir = os.path.join("..", "data", "embeddings")
images_dir = r"C:\Users\91939\OneDrive\Desktop\fashion\data\catlog\images"  # 👈 your folder path

# Load FAISS index + filenames
index = faiss.read_index(os.path.join(embed_dir, "faiss_index.bin"))
filenames = np.load(os.path.join(embed_dir, "filenames.npy"))

def get_similar_images(query_image, k=5):
    """Return top k visually similar image filenames"""
    img = Image.open(query_image).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype("float32")

    _, indices = index.search(emb, k)
    similar_files = [filenames[i] for i in indices[0]]
    return similar_files

if __name__ == "__main__":
    query_image = r"C:\Users\91939\OneDrive\Desktop\fashion\images\10095.jpg"  # 👈 test file
    results = get_similar_images(query_image)
    print("🔍 Top similar images:")
    for r in results:
        print("➡", r)
