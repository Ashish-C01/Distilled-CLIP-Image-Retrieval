import chromadb
from chromadb.config import Settings
import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image
import logging
import os

try:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    chroma_client = chromadb.PersistentClient(
        path='data', settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name='images')
    logger.info("ChromaDB loaded")

    device = 'cpu'
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = torch.nn.Linear(1024, 768)
    model.load_state_dict(torch.load(
        'mobilenet_v3_small_distilled_new_state_dict.pth', map_location=device))
    model.eval().to(device)
    logger.info("MobileNet loaded")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    def get_image_embedding(image):
        image = Image.open(
            f'images/{image}').convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            student_embedding = model(input_tensor)

        return torch.nn.functional.normalize(student_embedding, p=2, dim=1)

    embedding_list = []
    file_names = []
    ids = []
    for i in os.listdir('images'):
        embedding = get_image_embedding(i)
        embedding_list.append(embedding.squeeze(0).numpy().tolist())
        file_names.append({'path': f'images/{i}', 'type': 'photo'})
        ids.append(i)

    collection.add(
        embeddings=embedding_list,
        ids=ids,
        metadatas=file_names
    )
    logger.info("Embeddings inserted into ChromDB")


except Exception as e:
    logger.info(f"Exception occured {e}")
