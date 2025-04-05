# Face Detection and Recognition Project

The project is a web application for detecting and recognizing faces in videos. Recognition is implemented using FaceNet, ensuring high accuracy with just one photo of the person of interest.

## Contents

1. [Project Structure](#project-structure)  
2. [Dependencies](#dependencies)  
3. [Setup and Installation](#setup-and-installation)  
4. [Site Description](#site-description)  
5. [Features](#features)  
6. [Example Usage](#example-usage)  
7. [Work Visualization](#work-visualization)  
8. [License](#license)

---

## Project Structure

```
/face_recognition_site
|-- examples/
|   |-- example.png  # Screenshot of the website in action
|   |-- example_video.m4 # Detection Video
|
|-- templates/
|   |-- index.html
|
|-- app.py # Flask app
|
|-- create_embeddings.py # Script for creating embeddings
|
|-- detection.py # Script for creating detection
```

---

## Dependencies

Before running the project, install the following dependencies:

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Facenet PyTorch
- Flask
- MoviePy
- Pandas
- Pillow
- Scikit-learn
- Werkzeug

Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. Clone or download the [Sort](https://github.com/abewley/sort) repository.
2. Clone or download this repository.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the path to [Sort](https://github.com/abewley/sort) in the `path_to_sort` variable.

---

## Site Description

The website enables face detection and recognition in videos. Users can upload photos of the people they want to recognize. If a matching face is detected in the video, it is labeled with the corresponding name.

---

## Features

- **Face Detection**: Automatically detects all faces in the frame.
- **Photo Upload for Recognition**: Users can upload photos of people they want to identify.
- **Saving Unknown Faces**: Faces not recognized among the uploaded photos are saved for later processing.
- **Face Identification**: Recognized faces are labeled with names in the video.
- **Labeling Unknown Faces**: Faces seen multiple times are labeled with a unique index.

---

## Example Usage

1. The user uploads photos of people. The photo filename should match the person’s name.
2. The website processes the photo using FaceNet and generates a file with embeddings. Uploaded photos are displayed on the site.
3. The user uploads a video.
4. The processed video is displayed on the website with detected faces labeled by name.

---

## Work Visualization

![Website Screenshot](examples/example.png)

<img src="examples/example.gif" width="1919">

---

## License

This project is open-source and distributed under the MIT License.
