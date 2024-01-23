import numpy as np
import cv2
import os
from ultralytics import YOLO
from keras.models import load_model
from PIL import Image
from scipy.stats import entropy


class TrashModel:
    def __init__(self, video):
        self.video = video
        self.dir = str(os.path.dirname(os.path.abspath(__file__)))

        self.sp_class_indices = ["cardboard", "fabric", "glass", "metal", "paper"]
        self.mn_class_indices = ["cardboard", "fabric", "glass", "metal", "paper"]

        self.detector_model = YOLO("yolov8m-seg.pt")

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
            result = self.detector_model(image_np)[0]
            num_of_objects = len(result.boxes)
            print("\nNum of objects detected: ", num_of_objects)

            if num_of_objects > 0:
                for i in range(0, num_of_objects):
                    cropped_obj = None

                    x1 = result.boxes.xyxy[i][0]
                    y1 = result.boxes.xyxy[i][1]
                    x2 = result.boxes.xyxy[i][2]
                    y2 = result.boxes.xyxy[i][3]

                    x1 = int(np.array(x1))
                    y1 = int(np.array(y1))
                    x2 = int(np.array(x2))
                    y2 = int(np.array(y2))

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
                entropy_main = self.shannon_entropy(predictions)

                # Check if the prediction is confident enough
                tmp1 = sorted(predictions)
                first_max = tmp1[-1]
                second_max = tmp1[-2]

                if (first_max - second_max >= 0.3) or (
                    first_max - second_max > 0.15 and entropy_main <= 0.6
                ):
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_label = self.mn_class_indices[predicted_class_index]
                    obj["class"] = predicted_class_label
                    classes.append(predicted_class_label)

                else:
                    print("Using support model")

                    predictions_sp = self.support_model.predict(
                        np.expand_dims(cropped_obj, axis=0)
                    )
                    predictions_sp = list(predictions_sp.tolist())[0]

                    tmp2 = sorted(predictions_sp)
                    first_max_sp = tmp2[-1]
                    second_max_sp = tmp2[-2]
                    entropy_support = self.shannon_entropy(predictions_sp)

                    print("Entropy main - support: ", entropy_main, entropy_support)

                    if (
                        (
                            entropy_main - entropy_support >= 0.2
                            and entropy_support < 0.6
                        )
                        or (first_max_sp - second_max_sp >= 0.3)
                        or (
                            first_max_sp - second_max_sp > 0.15
                            and entropy_support <= 0.6
                        )
                    ):
                        predicted_class_index = np.argmax(predictions_sp)
                        predicted_class_label = self.sp_class_indices[
                            predicted_class_index
                        ]
                        obj["class"] = predicted_class_label
                        classes.append(predicted_class_label)
                        
                    else:
                        diff_top_main = first_max - second_max
                        diff_top_support = first_max_sp - second_max_sp

                        weight_main = (1 - entropy_main) * diff_top_main
                        weight_support = (1 - entropy_support) * diff_top_support

                        print("Weight main - support: ", weight_main, weight_support)

                        weighted_predictions = []
                        for i in range(5):
                            weighted_predictions.append(
                                weight_main
                                * predictions[i]
                                * (1 - self.shannon_entropy(predictions))
                                + weight_support
                                * predictions_sp[i]
                                * (1 - self.shannon_entropy(predictions_sp))
                            )
                        print("Weighted predictions: ", weighted_predictions)

                        predicted_class_index = np.argmax(weighted_predictions)
                        predicted_class_label = self.mn_class_indices[
                            predicted_class_index
                        ]
                        obj["class"] = predicted_class_label
                        classes.append(predicted_class_label)

            except ValueError as error:
                print("Value error: ", error)
                continue

            except TypeError as error:
                print("Type error: ", error)
                continue

            except RuntimeError as error:
                print("Runtime error: ", error)
                continue

        return classes

    def shannon_entropy(self, predictions):
        probabilities = np.asarray(predictions)
        probabilities = probabilities / np.sum(probabilities)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
