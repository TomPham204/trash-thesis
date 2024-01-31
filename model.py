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

    def get_color_areas(self, sub_image, tolerate=20, diff=30):
        # gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        # print("Finished converting to gray")

        # image_entropy = self.image_entropy(gray_image)

        image_entropy = self.calculate_entropy_rgb(sub_image)
        print("Image entropy: ", image_entropy)

        if image_entropy < 6:
            return False
        else:
            return True

        # max_val = np.max(gray_image)
        # min_val = np.min(gray_image)

        # if max_val - min_val < diff:
        #     return False

        # area_max = 0
        # area_min = 0

        # for i in range(gray_image.shape[0]):
        #     for j in range(gray_image.shape[1]):
        #         if gray_image[i, j] > max_val - tolerate:
        #             area_max += 1
        #         if gray_image[i, j] < min_val + tolerate:
        #             area_min += 1

        # print('Finished calculating areas')

        # if (
        #     max_val - min_val > diff
        #     and abs(area_max - area_min) / ((area_max + area_min) / 2) < 0.8
        # ):
        #     return True
        # else:
        #     return False

    def enhance_detection(self, image_np):
        height, width, _ = image_np.shape
        sub_height, sub_width = height // 3, width // 3
        tmp_segment=[]

        for i in range(3):
            for j in range(3):
                sub_image = image_np[
                    i * sub_height : (i + 1) * sub_height,
                    j * sub_width : (j + 1) * sub_width,
                ]
                if self.get_color_areas(sub_image, 20, 30):
                    tmp_segment.append({"image": sub_image, "class": ""})
        return tmp_segment

    def segment_objects(self, source, isEnhanced):
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
                
                if isEnhanced:
                    tmp_segment=self.enhance_detection(image_np)
                    segmented_objects.extend(tmp_segment)

            else:
                # YOLO failed to detect any objects, switch to enhanced detection
                tmp_segment=self.enhance_detection(image_np)
                segmented_objects.extend(tmp_segment)

                if len(segmented_objects) == 0:
                    segmented_objects.append({"image": image_np, "class": ""})

        except Exception as error:
            print("Error: ", error)
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

            except Exception as error:
                print("Error: ", error)
                continue

        return classes

    def shannon_entropy(self, predictions):
        probabilities = np.asarray(predictions)
        probabilities = probabilities / np.sum(probabilities)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def image_entropy(self, image):
        # Calculate the histogram of the image
        histogram = np.histogram(image, bins=256)[0]

        # Normalize the histogram
        histogram = list(filter(lambda p: p > 0, histogram / np.sum(histogram)))

        # Calculate the entropy
        entropy = -np.sum(np.multiply(histogram, np.log2(histogram)))

        return entropy

    def calculate_entropy_rgb(self, image):
        # Split the image into its color channels
        r, g, b = cv2.split(image)

        # Calculate the entropy for each color channel
        entropy_r = self.image_entropy(r)
        entropy_g = self.image_entropy(g)
        entropy_b = self.image_entropy(b)

        # Combine the entropies in some way (here we take the average)
        entropy = (entropy_r + entropy_g + entropy_b) / 3

        return entropy
