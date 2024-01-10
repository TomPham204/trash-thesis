import tkinter as tk
import cv2
from UI import TrashUI
from model import TrashModel
import threading


def process_camera_frame(root, trash_model, video):
    while True:
        trash_ui = TrashUI.getInstance(root, video)
        source = trash_ui.getCurrentSource()

        segmented_objects = trash_model.segment_objects(source)

        trash_ui.update_segmented_objects_preview(segmented_objects)
        classes = trash_model.predict_classes(segmented_objects)
        trash_ui.update_classes_list_preview(classes)

        if source == "live_feed":
            threading.Event().wait(2)
        else:
            threading.Event().wait(0.2)


def main():
    root = tk.Tk()
    root.minsize(720, 480)
    video = cv2.VideoCapture(0)
    trash_ui = TrashUI.getInstance(root, video)
    trash_model = TrashModel(video)

    ui_thread = threading.Thread(target=trash_ui.update_source_preview, daemon=True)
    ui_thread.start()

    process_frame_thread = threading.Thread(
        target=process_camera_frame,
        args=(root, trash_model, video),
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
