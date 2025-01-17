import sys
path_to_sort = 'C:/Users/nikso/sort'
sys.path.append(path_to_sort)
from sort import Sort  # SORT
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip
from moviepy.video.fx import crop
from moviepy.video.fx.resize import resize

def process_video(embeddings_path, video_path, output_path, output_path_with_audio):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    try:
        employee_embeddings = pd.read_csv(embeddings_path)
        employee_embeddings['Embedding'] = employee_embeddings['Embedding'].apply(
            lambda x: np.fromstring(x, sep=',')
        )
    except FileNotFoundError:
        employee_embeddings = pd.DataFrame(columns=['Name', 'Embedding'])

    tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.2)

    # Dictionary to store track embeddings
    track_embeddings = {}

    # Dictionary for unknown faces
    unknown_embeddings = {}

    # Video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

        detections = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                prob = probs[i]
                if prob > 0.9:  # Probability threshold filtering
                    detections.append([x1, y1, x2, y2, prob])

        # Update tracker
        if len(detections) == 0:
            trackers = np.empty((0, 5))  # Empty array if no detections
        else:
            trackers = tracker.update(np.array(detections))

        active_tracks = set()  # To track active tracks

        for track in trackers:
            x1, y1, x2, y2, track_id = map(int, track)
            active_tracks.add(track_id)

            # Check if the track is in the dictionary
            if track_id in track_embeddings and frame_count % 15 != 0:
                name = track_embeddings[track_id]['name']
            else:
                # Extract the face
                aligned_face = mtcnn.extract(image, [[x1, y1, x2, y2]], save_path=None)
                if aligned_face is not None:
                    aligned_face_tensor = aligned_face.to(device)

                    # Calculate embedding
                    embedding = facenet(aligned_face_tensor)
                    embedding = embedding.cpu().detach().numpy().flatten()

                    # Compare with employee database
                    similarities = []
                    for index, row in employee_embeddings.iterrows():
                        stored_embedding = row['Embedding']
                        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                        similarities.append((row['Name'], similarity))

                    # Compare with unknown faces
                    for name, unknown_embed in unknown_embeddings.items():
                        similarity = cosine_similarity([embedding], [unknown_embed])[0][0]
                        similarities.append((name, similarity))

                    # Find the most similar face
                    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                    
                    try:
                        if similarities[0][1] <= 0.4:  # Similarity threshold
                            name = f"Unknown_{track_id}"
                            if name not in unknown_embeddings:
                                unknown_embeddings[name] = embedding
                        else:
                            name = similarities[0][0]
                    except IndexError:
                        name = f"Unknown_{track_id}"
                        if name not in unknown_embeddings:
                                unknown_embeddings[name] = embedding

                    # Save to dictionary
                    track_embeddings[track_id] = {'embedding': embedding, 'name': name}

            # Draw rectangle and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Remove tracks that are no longer active
        inactive_tracks = set(track_embeddings.keys()) - active_tracks
        for track_id in inactive_tracks:
            del track_embeddings[track_id]
        
        out.write(frame)

    cap.release()
    out.release()

    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    output_video = VideoFileClip(output_path)

    if output_video.size[0] > output_video.size[1]:  # Horizontal video
        resized_video = resize(output_video, width=1920)
        final_video = resized_video.crop(width=1920, height=1080)
    else:  # Vertical video
        resized_video = resize(output_video, height=1080)
        final_video = resized_video.on_color(
            size=(1920, 1080), color=(0, 0, 0), pos=("center", "center")
        )
    
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(output_path_with_audio, codec="libx264", audio_codec="aac")
