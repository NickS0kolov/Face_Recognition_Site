from flask import Flask, request, render_template, send_from_directory, session
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory
import shutil
import uuid
import threading

from detection import process_video
from create_embeddings import create_embeddings

app = Flask(__name__)
app.secret_key = os.urandom(24) 

def generate_user_id():
    return str(uuid.uuid4())

@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = generate_user_id()

    global UPLOAD_FOLDER_PHOTOS, UPLOAD_FOLDER_VIDEOS, PROCESSED_FOLDER, EMBEDDINGS_FILE, USER_ID_PATH
    USER_ID_PATH = str(session.get('user_id')) + "_FOLDER"
    UPLOAD_FOLDER_PHOTOS = os.path.join(USER_ID_PATH, 'uploads/photos')
    UPLOAD_FOLDER_VIDEOS = os.path.join(USER_ID_PATH, 'uploads/videos')
    PROCESSED_FOLDER = os.path.join(USER_ID_PATH, 'processed')
    EMBEDDINGS_FILE = os.path.join(USER_ID_PATH, 'employee_embeddings.csv')

    os.makedirs(UPLOAD_FOLDER_PHOTOS, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER_VIDEOS, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    user_id = str(session.get('user_id'))
    threading.Timer(600, delete_folder_later, [user_id]).start()

def delete_folder_later(user_id):
    user_folder = f"{user_id}_FOLDER"
    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        print(f"Папка {user_folder} удалена")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/photos/<filename>')
def uploaded_photo(filename):
    return send_from_directory(UPLOAD_FOLDER_PHOTOS, filename)

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    files = request.files.getlist('photo')
    uploaded_photos = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_PHOTOS, filename)
            file.save(filepath)
            
            name = os.path.splitext(filename)[0]
            uploaded_photos.append({'name': name, 'filename': filename})

    global employee_embeddings_df
    employee_embeddings_df = create_embeddings(EMBEDDINGS_FILE, UPLOAD_FOLDER_PHOTOS)

    return render_template('index.html', uploaded_photos=uploaded_photos)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER_VIDEOS, filename)
        file.save(filepath)

        process_video(EMBEDDINGS_FILE, filepath, os.path.join(PROCESSED_FOLDER, 'output_video.mp4'), os.path.join(PROCESSED_FOLDER, 'output_video_with_audio.mp4'))
    
    try:
        uploaded_photos = [{'name': row['Name'], 'filename': row['Name'] + '.jpg'} for _, row in employee_embeddings_df.iterrows()]
        return render_template('index.html', process_video_file=filepath, uploaded_photos = uploaded_photos)
    except NameError:
        return render_template('index.html', process_video_file=filepath)

@app.route('/processed_video')
def processed_video():
    return send_from_directory(PROCESSED_FOLDER, 'output_video_with_audio.mp4')


if __name__ == '__main__':
    app.run(debug=True)