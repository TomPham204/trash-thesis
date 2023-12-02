import numpy as np
import cv2
import os
from imageai.Detection import ObjectDetection
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
from PIL import Image


class TrashModel:
    def __init__(self, video):
        self.video = video
        self.dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.class_indices = {
            "cardboard": 0,
            "fabric": 1,
            "glass": 2,
            "metal": 3,
            "paper": 4,
        }
        self.class_names = ["cardboard", "paper", "fabric", "glass", "metal"]

        # Initialize ObjectDetection from ImageAI for segmenting objects
        self.detector_model = ObjectDetection()
        self.detector_model.setModelTypeAsRetinaNet()
        self.detector_model.setModelPath(os.path.join(self.dir, "retina.pth"))
        self.detector_model.loadModel()

        # Initialize MobileNetV2 for trash classification
        self.predictor_model = load_model("keras_Model.h5", compile=False)

    def segment_objects(self):
        ret, frame = self.video.read()
        pil_image = Image.fromarray(frame)
        image_np = np.array(pil_image)

        _, detections = self.detector_model.detectObjectsFromImage(
            input_image=image_np, output_type="array"
        )

        segmented_objects = []

        if len(detections) > 0:
            for obj in detections:
                [x1, y1, x2, y2] = obj["box_points"]
                cropped_obj = frame[y1:y2, x1:x2]
                segmented_objects.append(
                    {"image": cropped_obj, "class": "To be predicted"}
                )
        return segmented_objects

    def predict_classes(self, segmented_objects):
        classes = []
        for obj in segmented_objects:
            cropped_obj = cv2.resize(obj["image"], (224, 224))
            cropped_obj = (cropped_obj.astype(np.float32) / 127.5) - 1

            predictions = self.predictor_model.predict(
                np.expand_dims(cropped_obj, axis=0)
            )

            # Map predictions to class labels
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = self.class_names[predicted_class_index]

            classes.append(predicted_class_label)
        return classes
