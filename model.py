import numpy as np
import cv2
import os

# from imageai.Detection import ObjectDetection
from ultralytics import YOLO
import tensorflow as tf
from keras.models import load_model
from PIL import Image


class TrashModel:
    def __init__(self, video):
        self.video = video
        self.dir = str(os.path.dirname(os.path.abspath(__file__)))

        self.sp_class_indices = ["cardboard", "fabric", "glass", "metal", "paper"]
        self.mn_class_indices = ["cardboard", "paper", "fabric", "glass", "metal"]

        # Initialize ObjectDetection from ImageAI for segmenting objects
        # self.detector_model = ObjectDetection()
        # self.detector_model.setModelTypeAsRetinaNet()
        # self.detector_model.setModelPath(os.path.join(self.dir, "retina.pth"))
        # self.detector_model.loadModel()

        self.detector_model = YOLO("yolov8m-seg.pt")

        # Initialize MobileNetV2 for trash classification
        self.predictor_model = load_model("main.h5", compile=False)
        self.support_model = load_model("support.h5", compile=False)

    def segment_objects(self, source):
        if source == "live_feed":
            try:
                ret, frame = self.video.read()
                pil_image = Image.fromarray(frame)
                image_np = np.array(pil_image)

                # _, detections = self.detector_model.detectObjectsFromImage(
                #     input_image=image_np, output_type="array"
                # )

                detections = self.detector_model(image_np)
                segmented_objects = []

                if len(detections) > 0:
                    for obj in detections:
                        x1 = int(obj.boxes.xyxy[0][0])
                        y1 = int(obj.boxes.xyxy[0][1])
                        x2 = int(obj.boxes.xyxy[0][2])
                        y2 = int(obj.boxes.xyxy[0][3])
                        # [x1, y1, x2, y2] = obj["box_points"]
                        # cropped_obj = frame[y1:y2, x1:x2]

                        if image_np.shape[2] == 3:
                            cropped_obj = image_np[y1:y2, x1:x2]
                            segmented_objects.append(
                                {"image": cropped_obj, "class": ""}
                            )
                        else:
                            cropped_obj = image_np[x1:x2, y1:y2]
                            segmented_objects.append(
                                {"image": cropped_obj, "class": ""}
                            )
                else:
                    segmented_objects.append({"image": frame, "class": ""})
                return segmented_objects
            except AttributeError as error:
                print(error)
                pass
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

                predictions = list(predictions.tolist())[0]

                # Check if the prediction is confident enough
                tmp1 = sorted(predictions)
                first_max = tmp1[-1]
                second_max = tmp1[-2]

                if first_max - second_max < 0.15 * second_max:
                    print("Using support model, ", tmp1)
                    predictions_sp = self.support_model.predict(
                        np.expand_dims(cropped_obj, axis=0)
                    )
                    predictions_sp = list(predictions_sp.tolist())[0]

                    # Check which model is more confident
                    tmp2 = sorted(predictions_sp)
                    first_max_sp = tmp2[-1]
                    second_max_sp = tmp2[-2]

                    [c, f, g, m, p] = predictions_sp
                    reordered_predictions_sp = [c, p, f, g, m]

                    weighted_predictions = []

                    if first_max_sp - second_max_sp < first_max - second_max:
                        for i in range(5):
                            weighted_predictions.append(
                                0.7 * predictions[i] + 0.3 * reordered_predictions_sp[i]
                            )
                    else:
                        for i in range(5):
                            weighted_predictions.append(
                                0.3 * predictions[i] + 0.7 * reordered_predictions_sp[i]
                            )

                    print("Main predictions: ", predictions)
                    print("Support predictions: ", reordered_predictions_sp)
                    print("Weighted predictions: ", weighted_predictions)

                    # Map predictions to class labels
                    predicted_class_index = np.argmax(weighted_predictions)
                    predicted_class_label = self.mn_class_indices[predicted_class_index]
                    obj["class"] = predicted_class_label
                    classes.append(predicted_class_label)

                else:
                    # Map predictions to class labels
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_label = self.mn_class_indices[predicted_class_index]
                    obj["class"] = predicted_class_label
                    classes.append(predicted_class_label)

            except ValueError as error:
                print("Value error: ", error)
                continue

            except TypeError as error:
                print("Type error: ", error)
                continue

        return classes
