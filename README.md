#AI-Powered Fashion Recommendation System
The AI-Powered Fashion Recommendation System suggests visually similar fashion products using advanced computer vision techniques.
It leverages OpenAI’s CLIP model to extract embeddings from images and FAISS (Facebook AI Similarity Search) to find similar items instantly.
A sleek Streamlit web interface lets users upload an image and see top outfit recommendations — like Pinterest or Myntra’s “shop similar looks.
| Component         | Technology             | Purpose                        |
| ----------------- | ---------------------- | ------------------------------ |
| **Model**         | OpenAI CLIP (ViT-B/32) | Extracts deep image embeddings |
| **Search Engine** | FAISS                  | Fast vector similarity search  |
| **Frameworks**    | PyTorch, NumPy         | Core ML computation            |
| **Frontend**      | Streamlit              | Interactive user interface     |
| **Utilities**     | PIL, tqdm, OpenCV      | Image preprocessing & tracking |

Project Workflow
graph TD
A[Fashion Images] --> B[Extract Embeddings with CLIP]
B --> C[Build FAISS Index]
C --> D[Search Similar Images]
D --> E[Streamlit Web App]


fashion-recommender/
│
├── data/
│   ├── catalog/images/        # fashion image dataset
│   └── embeddings/           
│
├── src/
│   ├── extract_embeddings.py
│   ├── build_faiss_index.py
│   ├── search_similar.py
│   ├── app.py
│
├── requirements.txt
└── README.md

Getting Started
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/ai-powered-fashion-recommendation-system.git
cd ai-powered-fashion-recommendation-system
2️⃣ Create Environment
conda create -n fashionai python=3.10
conda activate fashionai
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add Your Dataset
Place your images inside:
data/catalog/images/
5️⃣ Extract Embeddings
python src/extract_embeddings.py
6️⃣ Build FAISS Index
python src/build_faiss_index.py
7️⃣ Run the Streamlit App
streamlit run src/app.py
