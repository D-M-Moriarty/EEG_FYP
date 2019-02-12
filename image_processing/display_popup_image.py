import tkinter
from tkinter import *

import PIL.ImageTk
import cv2
from PIL import Image


class DisplayImage:

    def __init__(self, image_path):
        self.window = None
        self.path = image_path

    def center_window(self, width, height):
        # get screen width and height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.window.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def create_window(self):
        # Create a window
        self.window = Tk()
        self.window.title("OpenCV and Tkinter")
        # Load an image using OpenCV
        cv_img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        height, width, no_channels = cv_img.shape
        self.center_window(width, height)
        # Create a canvas that can fit the above image
        canvas = Canvas(self.window, width=width, height=height)
        canvas.pack()
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        # Add a PhotoImage to the Canvas
        canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
        # Run the window loop
        self.window.mainloop()


# DisplayImage("/Users/darrenmoriarty/ml/EEG_FYP/image_processing/images/arrow_down0.4.png").create_window()
