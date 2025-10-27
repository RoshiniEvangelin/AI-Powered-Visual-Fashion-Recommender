"""
01_extract_embeddings.py
------------------------
Extracts image embeddings using OpenAI's CLIP (or fallback openai-clip).
Saves embeddings and filenames in ../data/embeddings/.
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------
# 1️⃣ Try to import CLIP
# ----------------------------
try:
    import clip  # official OpenAI CLIP
    print("✅ Using official OpenAI CLIP library")
except ModuleNotFoundError:
    try:
        import openai_clip as clip  # fallback mirror
        print("✅ Using fallback openai-clip library")
    except ModuleNotFoundError:
        raise ImportError(
            "❌ Neither 'clip' nor 'openai_clip' is installed.\n"
            "Run one of the following:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  OR\n"
            "  pip install openai-clip"
        )

# ----------------------------
# 2️⃣ Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

# Directory setup
data_dir = r"C:\Users\91939\OneDrive\Desktop\fashion\data\catlog\images"

embed_dir = os.path.join("..", "data", "embeddings")
os.makedirs(embed_dir, exist_ok=True)

# ----------------------------
# 3️⃣ Process Images
# ----------------------------
embeddings = []
filenames = []

print("🔍 Extracting embeddings from images...")

for filename in tqdm(os.listdir(data_dir)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(data_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
                emb = emb.cpu().numpy()
            embeddings.append(emb)
            filenames.append(filename)
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to error: {e}")

# ----------------------------
# 4️⃣ Save Embeddings
# ----------------------------
if embeddings:
    embeddings = np.concatenate(embeddings)
    np.save(os.path.join(embed_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(embed_dir, "filenames.npy"), np.array(filenames))
    print(f"✅ Saved {len(filenames)} embeddings to {embed_dir}")
else:
    print("❌ No valid images found! Check your ../data/images/ folder.")
