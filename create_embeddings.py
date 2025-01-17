from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os


def create_embeddings(embedding_file_path, folder_path):
    try:
        employee_embeddings_df = pd.read_csv(embedding_file_path)
    except FileNotFoundError:
        employee_embeddings_df = pd.DataFrame(columns=['Name', 'Embedding'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    new_embeddings = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            image = Image.open(file_path)
        except Exception as e:
            print(f"Error opening file {filename}: {e}")
            continue

        name = filename.split('.')[0]
        if name not in employee_embeddings_df['Name'].to_list():
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

            if boxes is not None:
                for box, landmark in zip(boxes, landmarks):
                    aligned_face = mtcnn.extract(image, [box], save_path=None)
                    
                    if aligned_face is not None:
                        aligned_face_tensor = aligned_face.to(device)
                        embedding = facenet(aligned_face_tensor)
                        embedding = embedding.cpu().detach().numpy().flatten()

                        embedding_str = np.array2string(embedding, separator=',')[1:-1]

                        new_embeddings.append({'Name': name, 'Embedding': embedding_str})

                print(f"Embedding for {name} successfully added.")
            else:
                print(f"No faces detected in the image {filename}.")
        else:
            print(f"{name} is already present in the file.")

    if new_embeddings:
        new_df = pd.DataFrame(new_embeddings)
        employee_embeddings_df = pd.concat([employee_embeddings_df, new_df], ignore_index=True)
        employee_embeddings_df.to_csv(embedding_file_path, index=False)
        print(f"Embeddings successfully saved to the file {embedding_file_path}.")
    else:
        print("No new embeddings to save.")

    return employee_embeddings_df