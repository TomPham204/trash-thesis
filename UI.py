import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2


class TrashUI:
    _instance = None

    @classmethod
    def getInstance(cls, root, video):
        if not cls._instance:
            cls._instance = TrashUI(root, video)
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, root, video):
        self.root = root
        self.root.title("Trash Demo by Cong Tuan")
        self.root.configure(background="black")
        self.app_closing = False
        self.video = video

        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.top_frame = tk.Frame(self.container)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        title = tk.Label(
            self.top_frame,
            text="A model for classification of waste based on deep learning and image processing techniques",
        )
        title.pack()
        author = tk.Label(self.top_frame, text="Pham Cong Tuan ITITIU19060")
        author.pack()

        self.left_frame = tk.Frame(self.container)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(self.container)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        h1 = tk.Label(self.left_frame, text="Input preview")
        h1.pack()

        self.top_left_frame = tk.Frame(self.left_frame, width=300, height=300)
        self.top_left_frame.pack(padx=10, pady=10)

        h2 = tk.Label(self.left_frame, text="Segmented objects")
        h2.pack()

        self.bottom_left_frame = tk.Frame(self.left_frame, width=300, height=300)
        self.bottom_left_frame.pack(padx=10, pady=10)

        self.enhanceButtonStatus = tk.IntVar(value=1)
        useEnhancedDetection = tk.Checkbutton(
            self.right_frame,
            text="Enhance detection",
            variable=self.enhanceButtonStatus,
            onvalue=1,
            offvalue=0,
        )
        useEnhancedDetection.pack()

        h3 = tk.Label(self.right_frame, text="Select source")
        h3.pack()

        self.top_right_frame = tk.Frame(self.right_frame, width=300, height=100)
        self.top_right_frame.pack(padx=10, pady=10)

        h4 = tk.Label(self.right_frame, text="Class list")
        h4.pack()

        self.bottom_right_frame = tk.Frame(self.right_frame, width=300, height=500)
        self.bottom_right_frame.pack(padx=10, pady=10)

        # radio button to choose source
        self.source_var = tk.StringVar()
        self.source_var.set("live_feed")
        self.file_path = "live_feed"

        live_feed_radio = tk.Radiobutton(
            self.top_right_frame,
            text="Live Feed",
            variable=self.source_var,
            value="live_feed",
            command=self.update_camera_preview,
        )
        live_feed_radio.pack(anchor=tk.W)
        use_image_radio = tk.Radiobutton(
            self.top_right_frame,
            text="Use Image",
            variable=self.source_var,
            value="use_image",
            command=self.update_browse_image_preview,
        )
        use_image_radio.pack(anchor=tk.W)

    def getCurrentSource(self):
        if self.source_var.get() == "live_feed":
            return "live_feed"
        else:
            return self.file_path

    def getEnhancedDetectionStatus(self):
        return self.enhanceButtonStatus.get() == 1

    def stopUI(self):
        self.app_closing = False
        cv2.destroyAllWindows()

    def update_source_preview(self):
        if self.source_var.get() == "live_feed":
            self.update_camera_preview()
        else:
            self.update_browse_image_preview()

    def update_browse_image_preview(self):
        for widget in self.top_left_frame.winfo_children():
            widget.destroy()

        file_path = filedialog.askopenfilename()

        if file_path:
            self.source_var.set("use_image")
            self.file_path = file_path

            img = Image.open(file_path)
            img.thumbnail((350, 350))
            img = ImageTk.PhotoImage(image=img)

            self.top_left_frame.label = tk.Label(self.top_left_frame, image=img)
            self.top_left_frame.label.config(image=img)
            self.top_left_frame.label.img = img
            self.top_left_frame.label.pack()

    def update_camera_preview(self):
        self.source_var.set("live_feed")
        ret, frame = self.video.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((350, 350))
            img = ImageTk.PhotoImage(image=img)

            if not hasattr(self.top_left_frame, "label"):
                self.top_left_frame.label = tk.Label(self.top_left_frame, image=img)
            self.top_left_frame.label.config(image=img)
            self.top_left_frame.label.img = img
            self.top_left_frame.label.pack()

            if not self.app_closing:
                self.top_left_frame.label.after(50, self.update_camera_preview)

    def update_segmented_objects_preview(self, list_of_objects):
        for widget in self.bottom_left_frame.winfo_children():
            widget.destroy()

        if len(list_of_objects) == 0:
            return

        max_columns = 5

        for idx, obj in enumerate(list_of_objects):
            obj_img = obj["image"]
            obj_img = cv2.resize(obj_img, (60, 60))
            img = Image.fromarray(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            label = tk.Label(self.bottom_left_frame, image=img)
            label.img = img

            predicted_class = obj["class"]
            class_label = tk.Label(self.bottom_left_frame, text=predicted_class)

            row = idx // max_columns
            column = idx % max_columns

            label.grid(row=row * 2, column=column)
            class_label.grid(row=row * 2 + 1, column=column)

    def update_classes_list_preview(self, list_of_classes):
        for widget in self.bottom_right_frame.winfo_children():
            widget.destroy()

        for idx, obj_class in enumerate(list(set(list_of_classes))):
            label = tk.Label(self.bottom_right_frame, text=obj_class)
            label.grid(row=idx, column=0)
