import PIL
import streamlit as st
import numpy as np
from models import FaceDetector, FaceLandmarkDetector, SelfieSegmentation
import configs


def _helper(model_type):

    if model_type == 'Face Detection':
        detector = FaceDetector()
    elif model_type == 'Face Landmarks':
        detector = FaceLandmarkDetector()
    elif model_type == 'Selfie Segmentation':
        detector = SelfieSegmentation()
    return detector



def main():
    # Setting page layout
    st.set_page_config(
        page_title="Cleft Lip and Palate",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Cleft Lip and Palate") #Main page heading
    st.sidebar.header("List of Available Tasks") # Sidebar

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Face Detection', 'Face Landmarks', 'Selfie Segmentation'])

    default_predicted_imgs = {'Face Detection': configs.DEFAULT_DETECT_IMAGE,
                              'Face Landmarks': configs.DEFAULT_LANDMARK_IMAGE,
                              'Selfie Segmentation': configs.DEFAULT_SEGMENT_IMAGE,
                              }


    st.sidebar.header("List of Available Data Formats")
    source_radio = st.sidebar.radio(
        "Select Source", configs.SOURCES_LIST)

    source_img = None
    # If image is selected
    if source_radio == configs.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)
        detector = _helper(model_type)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(configs.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image, caption="Default Image")#, use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    uploaded_image = np.array(uploaded_image)
                    st.image(source_img, caption="Uploaded Image")#, use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(default_predicted_imgs[model_type])
                # default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image')#, use_column_width=True)
            else:
                if st.sidebar.button('Predict'):
                    try:
                        if model_type == 'Face Stylization':
                            annotated_image = detector.predict(uploaded_image)
                        else:
                            res = detector.predict(uploaded_image)
                            annotated_image = detector.visualize(uploaded_image, res)
                        st.image(annotated_image, caption='Detected Image')
                                 #use_column_width=True)

                    except Exception as ex:
                        st.write("No image is uploaded yet!")

    else:
        st.error("Please select a valid source type!")


if __name__ == '__main__':
    main()