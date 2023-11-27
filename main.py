import tkinter as tk
import cv2
from UI import TrashClassificationUI
from model import TrashModel
import threading


def process_camera_frame(root, trash_ui, trash_model):
    segmented_objects = trash_model.segment_objects()
    trash_ui.update_segmented_objects_preview(segmented_objects)

    classes = trash_model.predict_classes(segmented_objects)
    trash_ui.update_classes_list_preview(classes)
    root.after(5000, process_camera_frame, root, trash_ui, trash_model)


def main():
    root = tk.Tk()
    video = cv2.VideoCapture(0)
    trash_ui = TrashClassificationUI(root, video)
    trash_model = TrashModel(video)

    ui_thread = threading.Thread(target=trash_ui.update_camera_preview, daemon=True)
    ui_thread.start()

    process_frame_thread = threading.Thread(
        target=process_camera_frame,
        args=(root, trash_ui, trash_model),
        daemon=True,
    )
    process_frame_thread.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        trash_ui.stopUI()
        video.release()


if __name__ == "__main__":
    main()
