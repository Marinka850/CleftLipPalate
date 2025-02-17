import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

st.title("DR Classifier")

upload_widget = st.file_uploader(label="Upload an image file", type=["png"])
if upload_widget is not None:
    #image_file = upload_widget.get()
    img = load_image(upload_widget)
    st.image(img, channels="RGB")
    #st.image(image, channels="RGB")

    # Load the pre-trained YOLO model
    yolo = YOLO('YOLOv8_classification_retina.pt')

    # Predict class for the uploaded image
    predictions = yolo.predict(img)
    class_name = predictions[0].probs.top5[0]

    # Write the predicted class to the Streamlit app
    st.write("Predicted class:", class_name)

#streamlit run retinaCloud.py