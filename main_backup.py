import cv2
from imageai.Detection import ObjectDetection
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import os

class_indices = {"cardboard": 0, "fabric": 1, "glass": 2, "metal": 3, "paper": 4}
app_closing = False
dir = str(os.path.dirname(os.path.abspath(__file__)))


def update_camera_preview():
    ret, frame = video.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(image=img)

        if hasattr(update_camera_preview, "label"):
            update_camera_preview.label.config(image=img)
            update_camera_preview.label.img = img
        else:
            update_camera_preview.label = tk.Label(top_left_frame, image=img)
            update_camera_preview.label.img = img
            update_camera_preview.label.pack()

        if not app_closing:
            update_camera_preview.label.after(40, update_camera_preview)


def update_segmented_objects_preview(list_of_objects):
    # Clear the bottom_left_frame before updating with new images
    for widget in bottom_left_frame.winfo_children():
        widget.destroy()

    # Display the segmented objects in the bottom_left_frame
    for idx, obj in enumerate(list_of_objects):
        obj_img = obj["image"]
        obj_img = cv2.resize(obj_img, (60, 60))
        img = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image=img)
        label = tk.Label(bottom_left_frame, image=img)
        label.img = img
        label.grid(row=0, column=idx)

    pass


def update_classes_list_preview(list_of_classes):
    # Clear the right_frame before updating with new classes
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Display the classes in the right_frame
    for idx, obj_class in enumerate(list_of_classes):
        label = tk.Label(right_frame, text=obj_class)
        label.grid(row=idx, column=0)
    pass


def process_frame():
    ret, frame = video.read()

    _, detections = detector_model.detectObjectsFromImage(
        input_image=frame, output_type="array"
    )

    if detections:
        segmented_objects = []
        for obj in detections:
            [x1, y1, x2, y2] = obj["box_points"]
            cropped_obj = frame[y1:y2, x1:x2]
            segmented_objects.append({"image": cropped_obj, "class": "To be predicted"})
        update_segmented_objects_preview(segmented_objects)

    if len(detections) > 0:
        classes = []
        for obj in detections:
            [x1, y1, x2, y2] = obj["box_points"]
            cropped_obj = frame[y1:y2, x1:x2]
            cropped_obj = cv2.resize(cropped_obj, (300, 300))
            predictions = predictor_model.predict(np.expand_dims(cropped_obj, axis=0))

            # Map predictions to class labels
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = [
                label
                for label, index in class_indices.items()
                if index == predicted_class_index
            ]
            predicted_class_label = (
                predicted_class_label[0] if predicted_class_label else "Unknown"
            )

            print("Detected class:", predicted_class_label)
            classes.append(predicted_class_label)

        if len(classes) > 0:
            update_classes_list_preview(classes)

    root.after(5000, process_frame)


####################################################################
predictor_model = MobileNetV2(
    classes=5,
    include_top=False,
    input_shape=(300, 300, 3),
)
x = predictor_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.4)(x)

predictions = Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(0.05))(
    x
)
predictor_model = tf.keras.models.Model(
    inputs=predictor_model.input, outputs=predictions
)
optim = SGD(learning_rate=0.002, momentum=0.9)
predictor_model.compile(
    optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"]
)
predictor_model.load_weights(os.path.join(dir, "weight.h5"))

# Initialize ObjectDetection from ImageAI for segmenting objects
detector_model = ObjectDetection()
detector_model.setModelTypeAsRetinaNet()
detector_model.setModelPath(os.path.join(dir, "retina.pth"))
detector_model.loadModel()

# Start capturing video from the laptop's camera
video = cv2.VideoCapture(0)

# Create UI
root = tk.Tk()
root.title("Trash Classification")
# root.attributes("-fullscreen", True)
root.configure(background="black")

container = tk.Frame(root)
container.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(container)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

top_left_frame = tk.Frame(left_frame, width=300, height=300)
bottom_left_frame = tk.Frame(left_frame, width=300, height=300)
right_frame = tk.Frame(container, width=300, height=600)

# camera preview is updated constantly, while segmented objects and classes are updated every 5 seconds
update_camera_preview()
top_left_frame.pack(padx=10, pady=10)
bottom_left_frame.pack(padx=10, pady=10)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

process_frame()
root.mainloop()

# close the program when exit
app_closing = True
video.release()
cv2.destroyAllWindows()
