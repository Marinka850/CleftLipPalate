# %%
import cv2
from typing import Tuple, Union
import math
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import configs
import streamlit as st


class FaceDetector:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path=configs.MODELS_DETECT)
        self.options = vision.FaceDetectorOptions(base_options=self.base_options)
        self.detector = vision.FaceDetector.create_from_options(self.options)

    def predict(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect(mp_image)


    def _normalized_to_pixel_coordinates(self,
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


    def visualize(self,
            image,
            detection_result
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
          image: The input RGB image.
          detection_result: The list of all "Detection" entities to be visualize.
        Returns:
          Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, configs.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                               width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (configs.MARGIN + bbox.origin_x,
                             configs.MARGIN + configs.ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        configs.FONT_SIZE, configs.TEXT_COLOR, configs.FONT_THICKNESS)

        return annotated_image


class FaceLandmarkDetector:
    def __init__(self, numFaces=1):
        self.base_options = python.BaseOptions(model_asset_path=configs.MODELS_LANDMARKS)
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=numFaces)

        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def predict(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect(mp_image)

    def visualize(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        return annotated_image


class SelfieSegmentation:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path=configs.MODELS_SEGMENT)
        self.options = vision.ImageSegmenterOptions(base_options=self.base_options,
                                       output_category_mask=True)
        self.detector = vision.ImageSegmenter.create_from_options(self.options)

    def predict(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.segment(mp_image)

    def visualize(self, image_data, segmentation_result):
        category_mask = segmentation_result.category_mask
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = configs.MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = configs.BG_COLOR
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        return output_image
