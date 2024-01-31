import tkinter as tk
import cv2
from UI import TrashUI
from model import TrashModel
import threading

video = None


def process_input(root, trash_model, video):
    while True:
        trash_ui = TrashUI.getInstance(root, video)
        source = trash_ui.getCurrentSource()
        isEnhanced= trash_ui.getEnhancedDetectionStatus()

        segmented_objects = trash_model.segment_objects(source, isEnhanced)

        classes = trash_model.predict_classes(segmented_objects)
        
        trash_ui.update_segmented_objects_preview(segmented_objects)
        trash_ui.update_classes_list_preview(classes)

        if source == "live_feed":
            threading.Event().wait(1.5)
        else:
            threading.Event().wait(0.2)


def attempt_to_get_video():
    global video

    try:
        video = cv2.VideoCapture(0)
    except Exception as e:
        print("Error: ", e)
        attempt_to_get_video()


def main():
    global video

    root = tk.Tk()
    root.minsize(640, 480)
    root.maxsize(1280, 720)

    attempt_to_get_video()

    trash_ui = TrashUI.getInstance(root, video)
    trash_model = TrashModel(video)

    ui_thread = threading.Thread(target=trash_ui.update_source_preview, daemon=True)
    ui_thread.start()

    process_frame_thread = threading.Thread(
        target=process_input,
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
