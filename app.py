from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fallback_dev_key')
# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the ML model (placeholder - replace with your actual model)
def load_model():
    try:
        model_path = 'model/attr_cm_model.pkl'
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                model, class_names = loaded
                return model, class_names
            else:
                return loaded, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# Preprocess the ECG image for the model
def preprocess_image(image_path):
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to the same size used in training (e.g., 224x224)
    img = cv2.resize(img, (42, 42))

    # Normalize pixel values (same as training)
    img = img / 255.0

    # Flatten to 1D array for XGBoost
    img_flat = img.flatten()

    # Return as a 2D array with shape (1, features) — batch format
    return np.expand_dims(img_flat, axis=0)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/causes')
def causes():
    return render_template('causes.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'ecg_image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['ecg_image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            try:
                model, class_names = load_model()
                if model:
                    processed_img = preprocess_image(filename)

                    # Flatten the image to 1D if using XGBoost
                    if len(processed_img.shape) > 1:
                        processed_img = processed_img.reshape(1, -1)

                    prediction = model.predict(processed_img)[0]
                    proba = model.predict_proba(processed_img)[0]

                    # Use provided class names if available
                    if class_names:
                        result = class_names.inverse_transform([prediction])[0]
                    else:
                        result = f"Class {prediction}"


                    confidence = np.max(proba) * 100

                    return render_template('result.html',
                                           filename=file.filename,
                                           result=result,
                                           confidence=f"{confidence:.2f}%")
                else:
                    flash('Model could not be loaded.')
                    return redirect(request.url)

            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)

    return render_template('upload.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)