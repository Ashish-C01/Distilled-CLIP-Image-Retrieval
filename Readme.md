# Distilled-CLIP-Image-Retrieval ([Live Application](https://huggingface.co/spaces/ashish-001/Image-Search-Using-Text))

This project is a text-to-image search application powered by a distilled MobileNetV3-Small model, leveraging a teacher-student distillation approach with OpenAI's CLIP as the teacher model. The application allows users to search for images using text descriptions and upload new images to expand the searchable database. It uses ChromaDB for efficient vector storage and Streamlit for an interactive web interface.

## Project Overview

- **Objective**: Enable efficient text-to-image search using a lightweight, distilled model trained on image-caption pairs.
- **Model**: MobileNetV3-Small, distilled from OpenAI CLIP (CLIP-ViT-Large-Patch14) to produce 768-dimensional embeddings.
- **Training**: The model was trained on the COCO 2014 training dataset, which contains over 82,000 images with captions, aligning image and text embeddings in a shared space.
- **Inference Speed**: MobileNetV3-Small reduces inference time by **98.35%** compared to CLIP, making it highly efficient for real-time applications.
- **Database**: Initially populated with images from the COCO 2017 validation dataset (5,000 images), stored in ChromaDB with precomputed embeddings.
- **Features**:
  - Search for images by entering text descriptions.
  - Upload new images, generate embeddings, and add them to the collection.
  - Display search results in a uniform 224x224 grid.
  - Sidebar options to upload images and adjust the number of results returned via a slider.
  - API deployed on Hugging Face to return text search embeddings using OpenAI-CLIP-ViT-Large-Patch14.
  - **Note**: Since the search model is distilled from CLIP, it may not achieve the same level of accuracy as the original CLIP model.

## Screenshot

Below is a screenshot of the application in action:

![Image1](<screenshot.png>)

## Files

- **`app.py`**: The Streamlit application for the text-to-image search and image upload interface.
- **`add_images.py`**: A standalone script to create a ChromaDB collection and insert embeddings of images present in the `images/` folder.
- **`mobilenet_v3_small_distilled_new_state_dict.pth`**: Pretrained weights for the distilled MobileNetV3-Small model.
- **`Performance_comparison.ipynb`**: Jupyter notebook for evaluating and comparing the performance of the distilled model against CLIP.
- **`CLIP_model_distillation.ipynb`**: Jupyter notebook containing the training process for distilling CLIP into MobileNetV3-Small.
- **`requirements.txt`**: Python dependencies required to run the project.
- **`README.md`**: This documentation file.

## Setting Up the Image and ChromaDB Directories

The `images/` and `chromadb/` directories are **not included** in the GitHub repository. To set them up:

1. **Create the directories manually:**
   ```sh
   mkdir images chromadb
   ```
2. **Populate `images/` with your dataset** (e.g., COCO 2017 validation images or custom images).
3. **Run `add_images.py` to generate embeddings and populate the ChromaDB collection:**
   ```sh
   python add_images.py
   ```

This process will store image embeddings inside the `chromadb/` directory, enabling the search functionality.

## Installation & Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Ashish-C01/Distilled-CLIP-Image-Retrieval.git
   cd Distilled-CLIP-Image-Retrieval
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   streamlit run app.py
   ```

## API Usage

An API has been deployed on Hugging Face to return the embeddings of search text using OpenAI-CLIP-ViT-Large-Patch14.

- **Endpoint**: `https://ashish-001-text-embedding-api.hf.space/embedding`
- **Example Request:**
  ```python
  import requests
  params = {"text": "a cat on a beach"}
  response = requests.get(
                'https://ashish-001-text-embedding-api.hf.space/embedding', params=params)
  print(response.json())
  ```
- **Response Format:**
  ```json
  {"embedding": [0.1, 0.2, 0.3, ...],"dimension":768}
  ```

## Results & Performance Evaluation

- MobileNetV3-Small significantly reduces inference time by **98.35%** compared to CLIP.
- Detailed analysis and comparisons are available in `Performance_comparison.ipynb`.

## Potential Improvements

- Improve model accuracy by distilling from a higher-capacity student model.
- Train on a more diverse dataset to improve generalization.
- Implement hybrid retrieval techniques combining embeddings with metadata filtering.

## Acknowledgments

- OpenAI for CLIP model.
- ChromaDB for vector storage.
- Streamlit for the interactive UI.

