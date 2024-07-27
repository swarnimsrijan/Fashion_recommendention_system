from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
#fdbsvbfgds
import pickle
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define and load the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
    if 'imagefile' not in request.files:
        return redirect(request.url)

    uploaded_file = request.files['imagefile']

    if uploaded_file.filename == '':
        return redirect(request.url)

    if uploaded_file:
        filename = uploaded_file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Recommendations
        indices = recommend(features, feature_list)

        # Get filenames of recommended images
        # recommended_filenames = [filenames[idx] for idx in indices[0][1:6]]

        # Make paths relative to static folder
        query_path = f'uploads/{filename}'
        recommended_paths = [f'uploads/{os.path.basename(filenames[idx])}' for idx in indices[0][1:6]]

        # Log paths for debugging
        print("Query Path:", query_path)
        print("Recommended Paths:", recommended_paths)

        return render_template('index.html', query_path=query_path, recommendations=recommended_paths)

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

if __name__ == "__main__":
    app.run(debug=True)


