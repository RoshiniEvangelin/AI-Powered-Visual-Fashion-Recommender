import streamlit as st
from PIL import Image
from search_similar import get_similar_images, images_dir
import os

st.title(" Fashion Visual Search")
uploaded = st.file_uploader("Upload an outfit image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=300)
    img.save("temp.jpg")
    results = get_similar_images("temp.jpg")
    st.subheader("Similar Outfits:")
    for r in results:
        st.image(os.path.join(images_dir, r), width=200)
