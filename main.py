import tkinter as tk
import cv2
from UI import TrashUI
from model import TrashModel
import threading
import time


class TrashApp:
    def __init__(self):
        self.video = None
        self.trash_ui = None
        self.trash_model = None
        self.root = tk.Tk()
        self.root.minsize(640, 480)
        self.root.maxsize(1280, 720)

    def process_input(self):
        while True:
            source = self.trash_ui.getCurrentSource()
            isEnhanced = self.trash_ui.getEnhancedDetectionStatus()

            segmented_objects = self.trash_model.segment_objects(source, isEnhanced)

            classes = self.trash_model.predict_classes(segmented_objects)

            self.trash_ui.update_segmented_objects_preview(segmented_objects)
            self.trash_ui.update_classes_list_preview(classes)

            if source == "live_feed":
                time.sleep(1.5)
            else:
                time.sleep(0.5)

    def get_video(self):
        if self.video is None:
            try:
                self.video = cv2.VideoCapture(1)
                # self.video = cv2.VideoCapture(0)
            except Exception as e:
                print("Error: ", e)
                self.get_video()

    def run(self):
        self.get_video()

        self.trash_ui = TrashUI.getInstance(self.root, self.video)
        self.trash_model = TrashModel(self.video)

        ui_thread = threading.Thread(
            target=self.trash_ui.update_source_preview, daemon=True
        )
        ui_thread.start()

        process_frame_thread = threading.Thread(
            target=self.process_input,
            daemon=True,
        )
        process_frame_thread.start()

        try:
            self.root.mainloop()
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self.trash_ui.stopUI()
            self.video.release()


if __name__ == "__main__":

    app = TrashApp()
    app.run()
