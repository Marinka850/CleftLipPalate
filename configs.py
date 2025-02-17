from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
# VIDEO = 'Video'
# WEBCAM = 'Webcam'
# RTSP = 'RTSP'
# YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE]

# Model Configs
MODELS_DIR = ROOT/ 'pretrained-models'
MODELS_DETECT = MODELS_DIR / 'blaze_face_short_range.tflite'
MODELS_LANDMARKS = MODELS_DIR / 'face_landmarker.task'
MODELS_SEGMENT = MODELS_DIR / 'selfie_segmenter.tflite'


# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image-12-_jpeg_jpg.rf.8da52d004d768452f10b13fcf079cc2a.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'image1_face_detection.jpg'
DEFAULT_LANDMARK_IMAGE = IMAGES_DIR / 'image1_face_landmarks.jpg'
DEFAULT_SEGMENT_IMAGE = IMAGES_DIR / 'image1_face_segmentation.jpg'


# # Videos config
# VIDEO_DIR = ROOT / 'videos'
# VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
# VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
# VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
# VIDEOS_DICT = {
#     'video_1': VIDEO_1_PATH,
#     'video_2': VIDEO_2_PATH,
#     'video_3': VIDEO_3_PATH,
# }

# # ML Model config
# MODEL_DIR = ROOT / 'weights'
# DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
# # SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# # Webcam
# WEBCAM_PATH = 0

# Detection Styles
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

# Segmentation style
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Page Layout Configuration
page_config = {

}