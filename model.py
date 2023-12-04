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
        self.class_names_tm = ["cardboard", "paper", "fabric", "glass", "metal"]

        # Initialize ObjectDetection from ImageAI for segmenting objects
        self.detector_model = ObjectDetection()
        self.detector_model.setModelTypeAsRetinaNet()
        self.detector_model.setModelPath(os.path.join(self.dir, "retina.pth"))
        self.detector_model.loadModel()

        # Initialize MobileNetV2 for trash classification
        self.predictor_model = load_model("main_model.h5", compile=False)
        self.support_model = load_model("weight.h5", compile=False)

    def segment_objects(self, source):
        if source == "live_feed":
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
            else:
                segmented_objects.append(
                    {"image": frame, "class": "To be predicted"}
                )
            return segmented_objects
        else:
            pil_image = Image.open(source)
            image_np = np.array(pil_image)

            _, detections = self.detector_model.detectObjectsFromImage(
                input_image=image_np, output_type="array"
            )

            segmented_objects = []

            if len(detections) > 0:
                for obj in detections:
                    [x1, y1, x2, y2] = obj["box_points"]
                    cropped_obj = image_np[y1:y2, x1:x2]
                    segmented_objects.append(
                        {"image": cropped_obj, "class": "To be predicted"}
                    )
            else:
                segmented_objects.append(
                    {"image": image_np, "class": "To be predicted"}
                )
            return segmented_objects

    def predict_classes(self, segmented_objects):
        classes = []
        for obj in segmented_objects:
            try:
                cropped_obj = cv2.resize(obj["image"], (224, 224))
                cropped_obj = (cropped_obj.astype(np.float32) / 127.5) - 1

                predictions = self.predictor_model.predict(
                    np.expand_dims(cropped_obj, axis=0)
                )
                print("here1", predictions)

                # compare max and second max to see if the prediction is confident enough
                tmp = predictions[0].argsort()[::-1]
                if predictions[0][tmp[0]] - predictions[0][tmp[1]] < 0.2:
                    predictions_sp = self.support_model.predict(
                        np.expand_dims(cropped_obj, axis=0)
                    )

                    # Reorder predictions_sp to align with the order of     predictions
                    reordered_predictions_sp = [0] * len(predictions[0])
                    original_order = np.argsort(tmp)
                    sp_order = [1, 5, 2, 3, 4]  # Order of predictions_sp

                    for idx, label_idx in enumerate(original_order):
                        reordered_predictions_sp[label_idx] = predictions_sp[0][
                            sp_order[idx] - 1
                        ]

                    # Perform weighted average
                    weighted_predictions = (
                        0.65 * predictions + 0.35 * np.array([reordered_predictions_sp])
                    ) / 2

                    # Map predictions to class labels
                    predicted_class_index = np.argmax(weighted_predictions)
                    predicted_class_label = self.class_names_tm[predicted_class_index]
                    print(predicted_class_label)

                    classes.append(predicted_class_label)
                else:
                    # Map predictions to class labels
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_label = self.class_names_tm[predicted_class_index]
                    print(predicted_class_label)

                    classes.append(predicted_class_label)
            except ValueError as error:
                print("Value error: ", error)
                continue
            except TypeError as error:
                print("Type error: ", error)
                continue
        return classes
