import chromadb
from chromadb.config import Settings
import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image
import logging
import streamlit as st
import requests
import json
import uuid
import os

try:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @st.cache_resource
    def load_mobilenet_model():
        device = 'cpu'
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = torch.nn.Linear(1024, 768)
        model.load_state_dict(torch.load(
            'mobilenet_v3_small_distilled_new_state_dict.pth', map_location=device))
        model.eval().to(device)
        return model

    @st.cache_resource
    def load_chromadb():
        chroma_client = chromadb.PersistentClient(
            path='data', settings=Settings(anonymized_telemetry=False))
        collection = chroma_client.get_collection(name='images')
        return collection

    model = load_mobilenet_model()
    logger.info("MobileNet loaded")
    collection = load_chromadb()
    logger.info("ChromaDB loaded")
    logger.info(
        f"Connected to ChromaDB collection images with {collection.count()} items")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    def get_image_embedding(image):
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = Image.open(image).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to('cpu')
        with torch.no_grad():
            student_embedding = model(input_tensor)

        return torch.nn.functional.normalize(student_embedding, p=2, dim=1).squeeze(0).tolist()

    def save_image(image_file):
        unique_filename = f"{image_file.name}"
        save_path = os.path.join('images', unique_filename)
        with open(save_path, "wb") as f:
            f.write(image_file.getbuffer())
        return save_path

    def resize_image(image_path, size=(224, 224)):
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            # Handle uploaded file
            img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(size, Image.LANCZOS)  # High-quality resizing
        return img_resized

    st.sidebar.header("Upload Images")
    image_files = st.sidebar.file_uploader(
        "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    num_images = st.sidebar.slider(
        "Number of results to return", min_value=1, max_value=10, value=3)

    if image_files:
        st.sidebar.subheader(
            "Add Images to collection")
        if st.sidebar.button("Add uploaded images"):
            for idx, image_file in enumerate(image_files):
                image_embedding = get_image_embedding(image_file)
                saved_path = save_image(image_file)
                unique_id = str(uuid.uuid4())
                metadata = {
                    'path': f'images/{image_file.name}', "type": "photo"
                }
                collection.add(
                    embeddings=[image_embedding],
                    ids=[unique_id],
                    metadatas=[metadata]
                )
                st.sidebar.success(
                    f"Image {image_file.name} added to the collection")

    st.title('Image Search Using  Text')
    st.write(
        "The images stored in this database are sourced from the [COCO 2017 Validation Dataset](https://cocodataset.org/#download).")
    st.write('Enter the text to search for images with matching description')
    text_input = st.text_input("Description", "Playground")
    if st.button("Search"):
        if text_input.strip():
            params = {'text': text_input}
            response = requests.get(
                'https://ashish-001-text-embedding-api.hf.space/embedding', params=params)
            if response.status_code == 200:
                logger.info("Embedding returned by API successfully")
                data = json.loads(response.content)
                embedding = data['embedding']
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=num_images
                )
                images = [results['metadatas'][0][i]['path']
                          for i in range(len(results['metadatas'][0]))]
                distances = [results['distances'][0][i]
                             for i in range(len(results['metadatas'][0]))]
                if images:
                    cols_per_row = 3
                    rows = (len(images)+cols_per_row-1)//cols_per_row
                    for row in range(rows):
                        cols = st.columns(cols_per_row)
                        for col_idx, col in enumerate(cols):
                            img_idx = row*cols_per_row+col_idx
                            if img_idx < len(images):
                                resized_img = resize_image(
                                    images[img_idx], size=(224, 224))
                                col.image(resized_img,
                                          caption=f"Image {img_idx+1}\ndistance {distances[img_idx]}", use_container_width=True)
                else:
                    st.write("No image found")
            else:
                st.write("Please try again later")
                logger.info(f"status code {response.status_code} returned")
        else:
            st.write("Please enter text in the text area")

except Exception as e:
    logger.info(f"Exception occured: {e}")
