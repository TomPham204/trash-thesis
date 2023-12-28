import numpy as np
import cv2
import os
from ultralytics import YOLO
from keras.models import load_model
from PIL import Image


class TrashModel:
    def __init__(self, video):
        self.video = video
        self.dir = str(os.path.dirname(os.path.abspath(__file__)))

        self.sp_class_indices = ["cardboard", "fabric", "glass", "metal", "paper"]
        self.mn_class_indices = ["cardboard", "fabric", "glass", "metal", "paper"]

        self.detector_model = YOLO("yolov8l-seg.pt")

        self.predictor_model = load_model("main.h5", compile=False)
        self.support_model = load_model("support.h5", compile=False)

    def segment_objects(self, source):
        segmented_objects = []
        pil_image = None

        if source == "live_feed":
            ret, frame = self.video.read()
            pil_image = Image.fromarray(frame)
        else:
            pil_image = Image.open(source)

        try:
            image_np = np.array(pil_image)
            detections = self.detector_model(image_np)

            if len(detections[0].boxes.xyxy) > 0:
                for obj in detections:
                    x1 = int(obj.boxes.xyxy[0][0])
                    y1 = int(obj.boxes.xyxy[0][1])
                    x2 = int(obj.boxes.xyxy[0][2])
                    y2 = int(obj.boxes.xyxy[0][3])
                    cropped_obj = None
                    if image_np.shape[2] == 3:
                        cropped_obj = image_np[y1:y2, x1:x2]
                    else:
                        cropped_obj = image_np[x1:x2, y1:y2]
                    segmented_objects.append({"image": cropped_obj, "class": ""})
            else:
                print("No objects detected")
                segmented_objects.append({"image": image_np, "class": ""})
        except AttributeError as error:
            print("AttributeError", error)
            pass
        except ValueError as error:
            print("ValueError", error)
            pass
        except TypeError as error:
            print("TypeError", error)
            pass
        except IndexError as error:
            print("IndexError", error)
            pass

        return segmented_objects

    def predict_classes(self, segmented_objects):
        classes = []
        for obj in segmented_objects:
            try:
                cropped_obj = cv2.resize(obj["image"], (224, 224)).astype(np.float32)

                predictions = self.predictor_model.predict(
                    np.expand_dims(cropped_obj, axis=0)
                )

                predictions = list(predictions.tolist())[0]
                # print(predictions)

                # Check if the prediction is confident enough
                tmp1 = sorted(predictions)
                first_max = tmp1[-1]
                second_max = tmp1[-2]

                if first_max - second_max < 0.1 * second_max:  # 8%
                    print("Using support model, ", tmp1)
                    predictions_sp = self.support_model.predict(
                        np.expand_dims(cropped_obj, axis=0)
                    )
                    predictions_sp = list(predictions_sp.tolist())[0]

                    R1 = max(predictions) // min(predictions)
                    R2 = max(predictions_sp) // min(predictions_sp)

                    # [c, f, g, m, p] = predictions_sp
                    # reordered_predictions_sp = [c, p, f, g, m]
                    reordered_predictions_sp = predictions_sp

                    weighted_predictions = []

                    for i in range(5):
                        weighted_predictions.append(
                            (R1 * 100 // (R1 + R2)) * predictions[i]
                            + (R2 * 100 // (R1 + R2)) * reordered_predictions_sp[i]
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

                print(predicted_class_label)

            except ValueError as error:
                print("Value error: ", error)
                continue

            except TypeError as error:
                print("Type error: ", error)
                continue

        return classes
