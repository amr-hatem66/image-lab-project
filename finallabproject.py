# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:01:00 2024

@author: amr
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.original_image = None
        self.current_image = None

        self.create_menu()
        self.create_canvas()
        self.create_controls()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

    def create_controls(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack()

        button_color = "#C3CED7"  # Updated button color

        tk.Button(controls_frame, text="Apply HPF", command=self.apply_hpf, bg=button_color).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Mean Filter", command=self.apply_mean_filter, bg=button_color).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Median Filter", command=self.apply_median_filter, bg=button_color).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Roberts Edge Detector", command=self.apply_roberts_edge_detector, bg=button_color).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Prewitt Edge Detector", command=self.apply_prewitt_edge_detector, bg=button_color).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Sobel Edge Detector", command=self.apply_sobel_edge_detector, bg=button_color).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Erosion", command=self.apply_erosion, bg=button_color).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Dilation", command=self.apply_dilation, bg=button_color).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Opening", command=self.apply_open, bg=button_color).grid(row=2, column=2, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Closing", command=self.apply_close, bg=button_color).grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Hough Circle Transform", command=self.apply_hough_circle_transform, bg=button_color).grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(controls_frame, text="Apply Low Pass Filter", command=self.apply_low_pass_filter, bg=button_color).grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    # Add buttons for segmentation
        tk.Button(controls_frame, text="Segment using Region Split", command=self.segment_region_split, bg=button_color).grid(row=6, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(controls_frame, text="Merge Segments using Thresholding", command=self.merge_segments_thresholding, bg=button_color).grid(row=7, column=0, columnspan=3, padx=5, pady=5)


    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.update_image()

    def update_image(self):
        if self.current_image is not None:
            image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.canvas.image = image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)

    def show_image_in_new_window(self, image, title="Processed Image"):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        canvas = tk.Canvas(new_window, width=image.width(), height=image.height())
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=image)
        new_window.mainloop()

    def apply_hpf(self):
        if self.original_image is not None:
            kernel_size = 5
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
            hpf_image = cv2.subtract(gray_image, blurred_image)
            hpf_image = cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR)
            self.current_image = hpf_image
            image = Image.fromarray(cv2.cvtColor(hpf_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "High Pass Filter")

    def apply_mean_filter(self):
        if self.original_image is not None:
            kernel_size = 5
            mean_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
            self.current_image = mean_image
            image = Image.fromarray(cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Mean Filter")

    def apply_median_filter(self):
        if self.original_image is not None:
            kernel_size = 3
            median_image = cv2.medianBlur(self.original_image, kernel_size)
            self.current_image = median_image
            image = Image.fromarray(cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Median Filter")

    def apply_roberts_edge_detector(self):
        if self.original_image is not None:
            roberts_image = cv2.Canny(self.original_image, 100, 200)
            self.current_image = roberts_image
            roberts_image = cv2.cvtColor(roberts_image, cv2.COLOR_GRAY2BGR)
            image = Image.fromarray(cv2.cvtColor(roberts_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Roberts Edge Detector")

    def apply_prewitt_edge_detector(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
            prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            prewitt_image = cv2.cvtColor(prewitt_image, cv2.COLOR_GRAY2BGR)
            self.current_image = prewitt_image
            image = Image.fromarray(cv2.cvtColor(prewitt_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Prewitt Edge Detector")

    def apply_sobel_edge_detector(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            sobel_image = cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR)
            self.current_image = sobel_image
            image = Image.fromarray(cv2.cvtColor(sobel_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Sobel Edge Detector")

    def apply_erosion(self):
        if self.original_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            erosion_image = cv2.erode(self.original_image, kernel, iterations=1)
            self.current_image = erosion_image
            image = Image.fromarray(cv2.cvtColor(erosion_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Erosion")

    def apply_dilation(self):
        if self.original_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            dilation_image = cv2.dilate(self.original_image, kernel, iterations=1)
            self.current_image = dilation_image
            image = Image.fromarray(cv2.cvtColor(dilation_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Dilation")

    def apply_open(self):
        if self.original_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            open_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            self.current_image = open_image
            image = Image.fromarray(cv2.cvtColor(open_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Opening")

    def apply_close(self):
        if self.original_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            close_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            self.current_image = close_image
            image = Image.fromarray(cv2.cvtColor(close_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Closing")

    def apply_hough_circle_transform(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.medianBlur(gray_image, 5)
            circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
            hough_image = self.original_image.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            self.current_image = hough_image
            image = Image.fromarray(cv2.cvtColor(hough_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Hough Circle Transform")

    def apply_low_pass_filter(self):
        if self.original_image is not None:
            low_pass_image = cv2.GaussianBlur(self.original_image, (15, 15), 0)
            self.current_image = low_pass_image
            image = Image.fromarray(cv2.cvtColor(low_pass_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Low Pass Filter")

    def segment_region_split(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            mean_value = np.mean(gray_image)
            _, thresholded_image = cv2.threshold(gray_image, mean_value, 255, cv2.THRESH_BINARY)
            thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
            self.current_image = thresholded_image
            image = Image.fromarray(cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Region Split Segmentation")

    def merge_segments_thresholding(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
            self.current_image = thresholded_image
            image = Image.fromarray(cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB))
            image = ImageTk.PhotoImage(image)
            self.show_image_in_new_window(image, "Thresholding Segmentation")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()