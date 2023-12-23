from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage,QClipboard
from PyQt5.QtCore import Qt 
import sys
import cv2
from matplotlib.pylab import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from traceback import print_exc 
from PyQt5.QtWidgets import QApplication, QMessageBox
import os 
from PIL import Image 
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PyQt5.QtCore import QFile,QTextStream

def butterworth_lpf(image, cutoff_frequency, order):
        rows, cols = image.shape
        mask = np.zeros((rows, cols), dtype=np.float32)
        center = (rows // 2, cols // 2)

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                mask[i, j] = 1 / (1 + (distance / cutoff_frequency)**(2 * order))

        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        f_transform_filtered = f_transform_shifted * mask
        f_transform_filtered_shifted = np.fft.ifftshift(f_transform_filtered)
        blurred_image = np.abs(np.fft.ifft2(f_transform_filtered_shifted))

        return blurred_image.astype(np.uint8)

class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("project.ui", self)
        style_sheet_file = QFile("design.css")
        style_sheet_file.open(QFile.ReadOnly | QFile.Text)
        style_loader = QTextStream(style_sheet_file)
        style_sheet = style_loader.readAll()
        style_sheet_file.close()
        self.setStyleSheet(style_sheet)
        self.show()
        self.exitButton.clicked.connect(self.exit_application)
        self.pushButton_18.clicked.connect(self.open_image)
        self.garybutton.clicked.connect(self.convert_to_grayscale)
        self.balckandwhite.clicked.connect(self.convert_to_black_and_white)
        self.complementButton.clicked.connect(self.convert_to_complement_image)
        self.pushButton_22.clicked.connect(self.apply_fourier_transform)
        self.save.clicked.connect(self.save_image)
        self.pushButton_23.clicked.connect(self.reset_image)


        #group 2 button 
        self.pushButton_13.clicked.connect(self.apply2_button_clicked)
        #group buttons 1
        self.pushButton_12.clicked.connect(self.apply_button_clicked)
        #group 3
        self.pushButton_14.clicked.connect(self.apply3_button_clicked)
        self.pushButton_17.clicked.connect(self.apply6_button_clicked)
        self.Addbutton.clicked.connect(self.add_value_to_image)
        self.subbutton.clicked.connect(self.sub_value_from_image)
        self.mulButton.clicked.connect(self.mul_value_to_image)
        self.divButton.clicked.connect(self.div_value_from_image)
        #log group
        self.pushButton_15.clicked.connect(self.apply4_button_clicked)
        #histogram branch
        self.pushButton_16.clicked.connect(self.apply5_button_clicked)

    #histogram group
    def apply5_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_9()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Histogram Equalization":
        
                    self.histogram_equalization()
                    
                elif selected_radio == "Histogram":
                    self.display_histogram_graph()
                elif selected_radio == "Histogram Streching":
                    self.histogram_stretching()

    def get_selected_radio_in_groupbox_9(self):
        group_box = self.findChild(QGroupBox, "groupBox_9")
        if group_box:
            for child in group_box.findChildren(QRadioButton):
                if child.isChecked():
                    print("Selected radio button:", child.text())  # Check which radio button is selected
                    return child.text()
        else:
            print("groupBox_9 not found")
        return None
    def histogram_equalization(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the original image to grayscale
                if len(self.original_image.shape) == 3:
                    # If the image has multiple channels, convert to grayscale
                    gray_image = Image.fromarray(self.original_image).convert('L')
                    image_array = np.array(gray_image)
                else:
                    # Image is already grayscale
                    image_array = np.array(self.original_image)

                # Check if the image is empty
                if image_array.size == 0:
                    raise ValueError("Image is empty.")

                # Check if the image has a valid shape
                if len(image_array.shape) != 2:
                    raise ValueError("Invalid image shape.")

                # Calculate the histogram of the image
                histogram = [0] * 256
                width, height = image_array.shape
                for y in range(height):
                    for x in range(width):
                        pixel_value = image_array[x, y]
                        histogram[pixel_value] += 1

                # Calculate the cumulative distribution function (CDF) of the histogram
                cdf = np.zeros(256, dtype=int)
                cdf[0] = histogram[0]
                for i in range(1, 256):
                    cdf[i] = cdf[i - 1] + histogram[i]

                # Normalize the CDF
                cdf_normalized = (cdf - cdf[0]) * 255 / (cdf[-1] - cdf[0])
                cdf_normalized = cdf_normalized.astype(int)

                # Apply histogram equalization to each pixel in the image
                for y in range(height):
                    for x in range(width):
                        pixel_value = image_array[y, x]
                        image_array[y, x] = cdf_normalized[pixel_value]

                # Convert NumPy array to QImage for display
                q_img = QImage(image_array.data, width, height, width, QImage.Format_Grayscale8)

                # Display the histogram-equalized image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def histogram_stretching(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the original image to grayscale
                if len(self.original_image.shape) == 3:
                    # If the image has multiple channels, convert to grayscale
                    gray_image = Image.fromarray(self.original_image).convert('L')
                    image_array = np.array(gray_image)
                else:
                    # Image is already grayscale
                    image_array = np.array(self.original_image)

                # Check if the image is empty
                if image_array.size == 0:
                    raise ValueError("Image is empty.")

                # Check if the image has a valid shape
                if len(image_array.shape) != 2:
                    raise ValueError("Invalid image shape.")

                # Get the desired minimum and maximum values from line edits
                min_value_str = self.lineEdit_3.text()
                max_value_str = self.lineEdit_4.text()

                # Validate and convert min and max values
                try:
                    min_value = int(min_value_str)
                    max_value = int(max_value_str)

                    # Check if min and max values are within valid range
                    if min_value < 0 or max_value > 255 or min_value >= max_value:
                        raise ValueError("Invalid min or max value. Ensure min < max and both are in the range [0, 255].")

                except ValueError:
                    raise ValueError("Invalid min or max value. Please enter valid integers.")

                # Perform histogram stretching for each pixel separately
                height, width = image_array.shape

                stretched_image = np.zeros_like(image_array, dtype=np.uint8)

                for y in range(height):
                    for x in range(width):
                        pixel_value = image_array[y, x]

                        # Apply histogram stretching formula manually
                        stretched_pixel_value = int((pixel_value - min_value) * (255.0 / (max_value - min_value)))

                        # Clip the pixel value to the valid range [0, 255]
                        stretched_pixel_value = stretched_pixel_value if stretched_pixel_value >= 0 else 0
                        stretched_pixel_value = stretched_pixel_value if stretched_pixel_value <= 255 else 255

                        # Update the pixel value in the stretched image
                        stretched_image[y, x] = stretched_pixel_value

                # Convert NumPy array to QImage for display
                q_img = QImage(stretched_image.data, width, height, width, QImage.Format_Grayscale8)

                # Display the stretched image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    #log group
    def apply4_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_8()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Log":
                    self.log_transformation()

                elif selected_radio == "Inverse Log":
                    self.inverse_log_transformation()


                elif selected_radio == "Gamma":
                    self.gamma_correction()
                
    def get_selected_radio_in_groupbox_8(self):
        group_box = self.findChild(QGroupBox, "groupBox_8")
        if group_box:
            for child in group_box.findChildren(QRadioButton):
                if child.isChecked():
                    print("Selected radio button:", child.text())  # Check which radio button is selected
                    return child.text()
        else:
            print("groupBox_8 not found")
        return None
    def log_transformation(self):
        if hasattr(self, 'original_image'):
            try:
                # Get constant value for log transformation from lineEdit_2
                constant_str = self.lineEdit_2.text()

                # Validate and convert the constant value
                try:
                    constant = float(constant_str)
                    if constant <= 0:
                        raise ValueError("Constant value must be greater than 0.")
                except ValueError:
                    raise ValueError("Invalid constant value. Please enter a valid number.")

                # Apply log transformation using numpy
                height, width, channels = self.original_image.shape
                transformed_image = np.zeros_like(self.original_image)

                for y in range(height):
                    for x in range(width):
                        for c in range(channels):
                            pixel_value = self.original_image[y, x, c]
                            transformed_pixel_value = int(constant * np.log(1 + pixel_value))
                            transformed_image[y, x, c] = transformed_pixel_value

                transformed_image = np.uint8(transformed_image)

                # Convert NumPy array to QImage for display
                q_img = QImage(transformed_image.data, width, height, 3 * width, QImage.Format_RGB888)

                # Display the log-transformed image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def inverse_log_transformation(self):
        if hasattr(self, 'original_image'):
            try:
                # Get constant value for inverse log transformation from lineEdit_2
                constant_str = self.lineEdit_2.text()

                # Validate and convert the constant value
                try:
                    constant = float(constant_str)
                    if constant <= 0:
                        raise ValueError("Constant value must be greater than 0.")
                except ValueError:
                    raise ValueError("Invalid constant value. Please enter a valid number.")

                # Apply inverse log transformation using numpy
                height, width, channels = self.original_image.shape
                inverse_transformed_image = np.zeros_like(self.original_image, dtype=np.uint8)

                for y in range(height):
                    for x in range(width):
                        for c in range(channels):
                            pixel_value = self.original_image[y, x, c]

                            # Round the result to handle large values
                            inverse_transformed_pixel_value = int(round(np.exp(pixel_value / constant) - 1))

                            # Clip the value to the valid uint8 range
                            inverse_transformed_pixel_value = np.clip(inverse_transformed_pixel_value, 0, 255)

                            inverse_transformed_image[y, x, c] = inverse_transformed_pixel_value

                # Convert NumPy array to QImage for display
                q_img = QImage(inverse_transformed_image.data, width, height, 3 * width, QImage.Format_RGB888)

                # Display the inverse log-transformed image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def gamma_correction(self):
        if hasattr(self, 'original_image'):
            try:
                # Get gamma value from lineEdit_2
                gamma_str = self.lineEdit_2.text()

                # Validate and convert gamma value
                try:
                    gamma = float(gamma_str)
                    if gamma <= 0:
                        raise ValueError("Gamma value must be greater than 0.")
                except ValueError:
                    raise ValueError("Invalid gamma value. Please enter a valid number.")

                # Apply gamma correction using nested loops
                height, width, channels = self.original_image.shape
                corrected_image = np.zeros_like(self.original_image)

                for y in range(height):
                    for x in range(width):
                        for c in range(channels):
                            pixel_value = self.original_image[y, x, c]
                            corrected_pixel_value = int((pixel_value / 255.0) ** (1.0 / gamma) * 255.0)
                            corrected_image[y, x, c] = corrected_pixel_value

                corrected_image = np.uint8(corrected_image)

                # Convert NumPy array to QImage for display
                q_img = QImage(corrected_image.data, width, height, 3 * width, QImage.Format_RGB888)

                # Display the gamma-corrected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
        #----------------------- group 1------------------------------------------
    def apply_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_5()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Average":
                    #print ("Average in if cond")
                    self.avrg_rdio_clicked()

                elif selected_radio == "Max":
                    self.max_rdio_clicked()

                elif selected_radio == "Min":
                    self.min_rdio_clicked()
                
                elif selected_radio == "Median":
                    self.med_rdio_clicked()

                elif selected_radio == "Ideal LPF":
                    self.idl_lpf_clicked()

                elif selected_radio == "ButterWorth LPF":
                    self.butter_clicked()

                elif selected_radio == "Gaussian LPF":
                    self.gaussian_Button_clicked()
            else:
                # If no radio button is selected, show a message box
                QMessageBox.warning(self, "No Radio Button Selected", "Please select a radio button!")
        
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def gaussian_Button_clicked(self):
        if hasattr(self, 'original_image'):
            # Retrieve the original image and perform Gaussian filtering
            image = cv2.imread(self.image_path)  # Read the original image
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian Blur
            # Convert the blurred image back to QPixmap for display
            height, width, _ = blurred_image.shape
            bytes_per_line = width * 3  # Assuming Format_RGB888

            # Ensure the data is properly formatted as bytes before creating the QImage
            bytes_data = blurred_image.data

            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def butter_clicked(self):
        cutoff_frequency = 20  # Modify this value accordingly
        order = 2  # Modify this value accordingly
        if hasattr(self, 'original_image'):
            filter_size = 3  # Define the filter size
            image = cv2.imread(self.image_path)  # Read the image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
            # Perform some processing on the gray_image using a Butterworth LPF
            # Example:
            blurred_image = butterworth_lpf(gray_image, cutoff_frequency, order)

            # Convert the blurred image back to a QImage for display
            height, width = blurred_image.shape[:2]
            bytes_per_line = width  # Assuming Format_Grayscale8

            # Ensure the data is properly formatted as bytes before creating the QImage
            bytes_data = blurred_image.tobytes()

            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    
    def idl_lpf_clicked(self):
        if hasattr(self, 'original_image'):
            cutoff_frequency = 50
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            try:
                f_transform = fft2(image)
                f_transform_shifted = fftshift(f_transform)
                rows, cols = image.shape
                mask = np.ones((rows, cols), dtype=np.float32)
                center = (rows // 2, cols // 2)

                for i in range(rows):
                    for j in range(cols):
                        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if distance >= cutoff_frequency:
                            mask[i, j] = 0

                f_transform_filtered = f_transform_shifted * mask
                f_transform_filtered_shifted = ifftshift(f_transform_filtered)

                sharpened_image = np.abs(ifft2(f_transform_filtered_shifted))
                sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

                height, width = sharpened_image.shape
                q_img = QImage(sharpened_image.tobytes(), width, height, width, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")



    def med_rdio_clicked(self):
        if hasattr(self, 'original_image'):
            filter_size = 3
            image = cv2.imread(self.image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            median_filtered_image = np.zeros((height, width), dtype=np.uint8)
            for x in range(height):
                for y in range(width):
                    start_x = max(0, x - filter_size // 2)
                    end_x = min(height, x + filter_size // 2 + 1)
                    start_y = max(0, y - filter_size // 2)
                    end_y = min(width, y + filter_size // 2 + 1)
                    region = gray_image[start_x:end_x, start_y:end_y]
                    median_value = np.median(region)
                    median_filtered_image[x, y] = median_value
            q_img_original = QImage(gray_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_original = QPixmap.fromImage(q_img_original)
            pixmap_original = pixmap_original.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_original)
            self.label_13.setAlignment(Qt.AlignCenter)

            q_img_filtered = QImage(median_filtered_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_filtered = QPixmap.fromImage(q_img_filtered)
            pixmap_filtered = pixmap_filtered.scaled(self.label_12.width(), self.label_12.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_filtered)
            self.label_12.setAlignment(Qt.AlignCenter)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def min_rdio_clicked(self):
        if hasattr(self, 'original_image'):
            filter_size = 3
            image = cv2.imread(self.image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            min_filtered_image = np.zeros((height, width), dtype=np.uint8)
            for x in range(height):
                for y in range(width):
                    start_x = max(0, x - filter_size // 2)
                    end_x = min(height, x + filter_size // 2 + 1)
                    start_y = max(0, y - filter_size // 2)
                    end_y = min(width, y + filter_size // 2 + 1)
                    region = gray_image[start_x:end_x, start_y:end_y]
                    min_value = np.min(region)
                    min_filtered_image[x, y] = min_value
            q_img_original = QImage(gray_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_original = QPixmap.fromImage(q_img_original)
            pixmap_original = pixmap_original.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_original)
            self.label_13.setAlignment(Qt.AlignCenter)

            q_img_filtered = QImage(min_filtered_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_filtered = QPixmap.fromImage(q_img_filtered)
            pixmap_filtered = pixmap_filtered.scaled(self.label_12.width(), self.label_12.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_filtered)
            self.label_12.setAlignment(Qt.AlignCenter)
        
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def avrg_rdio_clicked(self):
        if hasattr(self, 'original_image'):
            filter_size = 3
            image = cv2.imread(self.image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            filter_kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)
            average_filtered_image = np.zeros((height, width), dtype=np.uint8)
            for x in range(height):
                for y in range(width):
                    start_x = max(0, x - filter_size // 2)
                    end_x = min(height, x + filter_size // 2 + 1)
                    start_y = max(0, y - filter_size // 2)
                    end_y = min(width, y + filter_size // 2 + 1)
                    region = gray_image[start_x:end_x, start_y:end_y]
                    filtered_value = np.sum(region * filter_kernel[:region.shape[0], :region.shape[1]])
                    filtered_value = np.clip(filtered_value, 0, 255)
                    average_filtered_image[x, y] = filtered_value.astype(np.uint8)
            q_img_original = QImage(gray_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_original = QPixmap.fromImage(q_img_original)
            pixmap_original = pixmap_original.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_original)
            self.label_13.setAlignment(Qt.AlignCenter)

            q_img_filtered = QImage(average_filtered_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_filtered = QPixmap.fromImage(q_img_filtered)
            pixmap_filtered = pixmap_filtered.scaled(self.label_12.width(), self.label_12.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_filtered)
            self.label_12.setAlignment(Qt.AlignCenter)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def max_rdio_clicked(self):
        if hasattr(self, 'original_image'):
            filter_size = 3
            image = cv2.imread(self.image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            max_filtered_image = np.zeros((height, width), dtype=np.uint8)
            for x in range(height):
                for y in range(width):
                    start_x = max(0, x - filter_size // 2)
                    end_x = min(height, x + filter_size // 2 + 1)
                    start_y = max(0, y - filter_size // 2)
                    end_y = min(width, y + filter_size // 2 + 1)
                    region = gray_image[start_x:end_x, start_y:end_y]
                    max_value = np.max(region)
                    max_filtered_image[x, y] = max_value
            q_img_original = QImage(gray_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_original = QPixmap.fromImage(q_img_original)
            pixmap_original = pixmap_original.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_original)
            self.label_13.setAlignment(Qt.AlignCenter)

            q_img_filtered = QImage(max_filtered_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap_filtered = QPixmap.fromImage(q_img_filtered)
            pixmap_filtered = pixmap_filtered.scaled(self.label_12.width(), self.label_12.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap_filtered)
            self.label_12.setAlignment(Qt.AlignCenter)
            
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

        #@-------------------------------------------------------------------------
#----------------------- group 2 ------------------------------------------
#--------------------------------------------------------------------------

    def apply2_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_6()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Point":
                    self.point_radio_Button_clicked()

                elif selected_radio == "Vertical":
                    self.vertical_radio_Button_clicked()

                elif selected_radio == "Horizontal":
                    self.Horizontal_radio_Button_clicked()
                
                elif selected_radio == "Left Diagonal":
                    self.left_diagonal_radioButton_clicked()

                elif selected_radio == "Right Diagonal":
                    self.right_diagonal_radioButton_clicked()

                elif selected_radio == "Ideal HPF":
                    #problem dosen't perform the correct mask
                    self.Ideal_HPF_radioButton_clicked()

                elif selected_radio == "ButterWorth HPF":
                    self.butter_radio_Button_clicked()

                elif selected_radio == "Gaussian HPF":
                    self.gaussian1_Button_clicked()
                    

                elif selected_radio == "Mid Point":
                    self.midpoint_Button_clicked()
                    
            else:
                # If no radio button is selected, show a message box
                QMessageBox.warning(self, "No Radio Button Selected", "Please select a radio button!")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")




    def get_selected_radio_in_groupbox_6(self):
        group_box = self.findChild(QGroupBox, "groupBox_6")
        if group_box:
            for child in group_box.findChildren(QRadioButton):
                if child.isChecked():
                    print("Selected radio button:", child.text())  # Check which radio button is selected
                    return child.text()
        else:
            print("groupBox_5 not found")
        return None
    

#----------------------------------------------------------------------------------    
#-----------------------------------group3-----------------------------------------
#----------------------------------------------------------------------------------
    def apply3_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_7()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Point":
                    self.point1_radio_Button_clicked()

                elif selected_radio == "Vertical":
                    self.vertical1_radio_Button_clicked()
                    

                elif selected_radio == "Horizontal":
                    self.horizontal1_radio_Button_clicked()
                    
                
                elif selected_radio == "Left Diagonal":
                    self.left_diagonal1_radio_Button_clicked()
                    

                elif selected_radio == "Right Diagonal":
                    self.right_diagonal1_radio_Button_clicked()
                    

                
                    
            else:
                # If no radio button is selected, show a message box
                QMessageBox.warning(self, "No Radio Button Selected", "Please select a radio button!")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def get_selected_radio_in_groupbox_7(self):
        group_box = self.findChild(QGroupBox, "groupBox_7")
        if group_box:
            for child in group_box.findChildren(QRadioButton):
                if child.isChecked():
                    print("Selected radio button:", child.text())  # Check which radio button is selected
                    return child.text()
        else:
            print("groupBox_7 not found")
        return None
    
#---------------------- point processing for edge detection --------------------------------
    
    def point1_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                image = self.original_image.copy()
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply Sobel operators to compute gradients
                sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

                # Compute the gradient magnitude
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

                # Normalize the gradient magnitude to the range [0, 255]
                normalized_gradient = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

                # Convert NumPy array to QImage for display
                height, width = normalized_gradient.shape
                q_img = QImage(normalized_gradient.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the edge-detected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

#---------------------- vertical processing for edge detection --------------------------------
    def vertical1_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                if len(self.original_image.shape) == 3:
                    original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    original_image_gray = self.original_image

                # Apply the vertical Sobel operator
                sobel_vertical = cv2.Sobel(original_image_gray, cv2.CV_64F, 0, 1, ksize=3)

                # Compute the absolute value of the gradient
                abs_gradient = np.abs(sobel_vertical)

                # Normalize the gradient to the range [0, 255]
                normalized_gradient = np.uint8(255 * abs_gradient / np.max(abs_gradient))

                # Convert NumPy array to QImage for display
                height, width = normalized_gradient.shape[:2]
                q_img = QImage(normalized_gradient.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the vertical edge-detected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

#---------------------- Horizontal processing for edge detection --------------------------------
    def horizontal1_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                if len(self.original_image.shape) == 3:
                    original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    original_image_gray = self.original_image

                # Apply the horizontal Sobel operator
                sobel_horizontal = cv2.Sobel(original_image_gray, cv2.CV_64F, 1, 0, ksize=3)

                # Compute the absolute value of the gradient
                abs_gradient = np.abs(sobel_horizontal)

                # Normalize the gradient to the range [0, 255]
                normalized_gradient = np.uint8(255 * abs_gradient / np.max(abs_gradient))

                # Convert NumPy array to QImage for display
                height, width = normalized_gradient.shape[:2]
                q_img = QImage(normalized_gradient.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the horizontal edge-detected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

#---------------------- Left-diagonal processing for edge detection --------------------------------
    def left_diagonal1_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                if len(self.original_image.shape) == 3:
                    original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    original_image_gray = self.original_image

                # Define a left diagonal Sobel operator
                sobel_left_diagonal = np.array([[1, 0, -1],
                                                [0, 0, 0],
                                                [-1, 0, 1]], dtype=np.float64)

                # Apply the left diagonal Sobel operator
                left_diagonal_gradient = cv2.filter2D(original_image_gray, cv2.CV_64F, sobel_left_diagonal)

                # Compute the absolute value of the gradient
                abs_gradient = np.abs(left_diagonal_gradient)

                # Normalize the gradient to the range [0, 255]
                normalized_gradient = np.uint8(255 * abs_gradient / np.max(abs_gradient))

                # Convert NumPy array to QImage for display
                height, width = normalized_gradient.shape[:2]
                q_img = QImage(normalized_gradient.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the left diagonal edge-detected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

 #---------------------- Right-diagonal processing for edge detection --------------------------------
    def right_diagonal1_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                if len(self.original_image.shape) == 3:
                    original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    original_image_gray = self.original_image

                # Define a right diagonal Sobel operator
                sobel_right_diagonal = np.array([[-1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, -1]], dtype=np.float64)

                # Apply the right diagonal Sobel operator
                right_diagonal_gradient = cv2.filter2D(original_image_gray, cv2.CV_64F, sobel_right_diagonal)

                # Compute the absolute value of the gradient
                abs_gradient = np.abs(right_diagonal_gradient)

                # Normalize the gradient to the range [0, 255]
                normalized_gradient = np.uint8(255 * abs_gradient / np.max(abs_gradient))

                # Convert NumPy array to QImage for display
                height, width = normalized_gradient.shape[:2]
                q_img = QImage(normalized_gradient.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the right diagonal edge-detected image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def point_radio_Button_clicked(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Retrieve the original image
            image = self.original_image.copy()

            # Define a sharpening kernel (Laplacian)
            kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

            # Apply the kernel for image sharpening
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Display the sharpened image
            height, width, _ = sharpened_image.shape
            bytes_per_line = width * 3  

            bytes_data = sharpened_image.data
            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def vertical_radio_Button_clicked(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Retrieve the original image
            image = self.original_image.copy()

            # Define a vertical sharpening kernel
            kernel = np.array([[0, -1, 0],
                            [0, 5, 0],
                            [0, -1, 0]])

            # Apply the kernel for vertical image sharpening
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Display the sharpened image
            height, width, _ = sharpened_image.shape
            bytes_per_line = width * 3  # Assuming Format_RGB888

            bytes_data = sharpened_image.data
            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def Horizontal_radio_Button_clicked(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Retrieve the original image
            image = self.original_image.copy()

            # Define a horizontal sharpening kernel
            kernel = np.array([[-1, 0, -1],
                            [5, 0, 5],
                            [-1, 0, -1]])

            # Apply the kernel for horizontal image sharpening
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Display the sharpened image
            height, width, _ = sharpened_image.shape
            bytes_per_line = width * 3  # Assuming Format_RGB888

            bytes_data = sharpened_image.data
            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def left_diagonal_radioButton_clicked(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Retrieve the original image
            image = self.original_image.copy()

            # Define a left diagonal sharpening kernel
            kernel = np.array([[-1, -1, 5],
                            [-1, 0, -1],
                            [5, -1, -1]])

            # Apply the kernel for left diagonal image sharpening
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Display the sharpened image
            height, width, _ = sharpened_image.shape
            bytes_per_line = width * 3  # Assuming Format_RGB888

            bytes_data = sharpened_image.data
            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def right_diagonal_radioButton_clicked(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Retrieve the original image
            image = self.original_image.copy()

            # Define a right diagonal sharpening kernel
            kernel = np.array([[5, -1, -1],
                            [-1, 0, -1],
                            [-1, -1, 5]])

            # Apply the kernel for right diagonal image sharpening
            sharpened_image = cv2.filter2D(image, -1, kernel)

            # Display the sharpened image
            height, width, _ = sharpened_image.shape
            bytes_per_line = width * 3  # Assuming Format_RGB888

            bytes_data = sharpened_image.data
            q_img = QImage(bytes_data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def Ideal_HPF_radioButton_clicked(self):
        if hasattr(self, 'original_image'):
            cutoff_frequency = 30
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            try:
                f_transform = fft2(image)
                f_transform_shifted = fftshift(f_transform)
                rows, cols = image.shape
                mask = np.ones((rows, cols), dtype=np.float32)
                center = (rows // 2, cols // 2)

                for i in range(rows):
                    for j in range(cols):
                        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if distance <= cutoff_frequency:
                            mask[i, j] = 0

                f_transform_filtered = f_transform_shifted * mask
                f_transform_filtered_shifted = ifftshift(f_transform_filtered)

                sharpened_image = np.abs(ifft2(f_transform_filtered_shifted))
                sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

                height, width = sharpened_image.shape
                q_img = QImage(sharpened_image.tobytes(), width, height, width, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def butter_radio_Button_clicked(self):
        if hasattr(self, 'original_image'):
            cutoff_frequency = 10
            order = 7
            image = self.original_image.copy()

            try:
                # Convert the image to grayscale if it's in color
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Compute the 2D Fourier Transform of the image
                f_transform = fft2(image)

                # Shift the zero frequency component to the center
                f_transform_shifted = fftshift(f_transform)

                # Get image dimensions
                rows, cols = image.shape

                # Create a mask for the high-pass filter
                mask = np.ones((rows, cols), dtype=np.float32)
                center = (rows // 2, cols // 2)

                for i in range(rows):
                    for j in range(cols):
                        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        mask[i, j] = 1 - 1 / (1 + (distance / cutoff_frequency)**(2 * order))

                # Apply the high-pass filter in the frequency domain
                f_transform_filtered = f_transform_shifted * mask

                # Shift the result back to the original position
                f_transform_filtered_shifted = ifftshift(f_transform_filtered)

                # Compute the inverse Fourier Transform to get the sharpened image
                sharpened_image = np.abs(ifft2(f_transform_filtered_shifted))

                # Normalize the pixel values to the range [0, 255]
                sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

                # Convert NumPy array to QImage for display
                height, width = sharpened_image.shape
                q_img = QImage(sharpened_image.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the sharpened image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def gaussian1_Button_clicked(self):
        if hasattr(self, 'original_image'):
            cutoff_frequency = 10
            image = self.original_image.copy()

            try:
                # Convert the image to grayscale if it's in color
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Compute the 2D Fourier Transform of the image
                f_transform = fft2(image)

                # Shift the zero frequency component to the center
                f_transform_shifted = fftshift(f_transform)

                # Get image dimensions
                rows, cols = image.shape

                # Create a mask for the high-pass filter (Gaussian)
                mask = np.zeros((rows, cols), dtype=np.float32)
                center = (rows // 2, cols // 2)

                for i in range(rows):
                    for j in range(cols):
                        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        mask[i, j] = 1 - np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))

                # Apply the high-pass filter in the frequency domain
                f_transform_filtered = f_transform_shifted * mask

                # Shift the result back to the original position
                f_transform_filtered_shifted = ifftshift(f_transform_filtered)

                # Compute the inverse Fourier Transform to get the sharpened image
                sharpened_image = np.abs(ifft2(f_transform_filtered_shifted))

                # Normalize the pixel values to the range [0, 255]
                sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

                # Convert NumPy array to QImage for display
                height, width = sharpened_image.shape
                q_img = QImage(sharpened_image.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the sharpened image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def midpoint_Button_clicked(self, filter_size=3, contrast_factor=1.5, brightness_factor=10):
        if hasattr(self, 'original_image'):
            image = self.original_image.copy()

            try:
                # Convert the image to grayscale if it's in color
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Compute the 2D Fourier Transform of the image
                f_transform = fft2(image)

                # Shift the zero frequency component to the center
                f_transform_shifted = fftshift(f_transform)

                # Get image dimensions
                rows, cols = image.shape

                # Create a mask for the mid-point filter
                mask = np.ones((rows, cols), dtype=np.float32)
                center = (rows // 2, cols // 2)

                # Compute midpoint value for local region
                local_region = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size ** 2)
                midpoint = np.median(local_region)

                # Apply the mid-point filter in the frequency domain
                mask[center[0] - filter_size // 2: center[0] + filter_size // 2 + 1,
                     center[1] - filter_size // 2: center[1] + filter_size // 2 + 1] = midpoint

                f_transform_filtered = f_transform_shifted * mask

                # Shift the result back to the original position
                f_transform_filtered_shifted = ifftshift(f_transform_filtered)

                # Compute the inverse Fourier Transform to get the sharpened image
                sharpened_image = np.abs(ifft2(f_transform_filtered_shifted))

                # Normalize the pixel values to the range [0, 255]
                sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

                # Adjust contrast and brightness
                enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=contrast_factor, beta=brightness_factor)

                # Convert NumPy array to QImage for display
                height, width = enhanced_image.shape
                q_img = QImage(enhanced_image.tobytes(), width, height, width, QImage.Format_Grayscale8)

                # Display the enhanced image using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QClipboard().instance()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")  

    def exit_application(self):
        QApplication.quit()
    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home', 'Image files (*.jpg *.png *.bmp)')

        if fname:
            pixmap = QPixmap(fname)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.label_12.width(), self.label_12.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_12.setPixmap(pixmap)
                self.label_12.setAlignment(Qt.AlignCenter)
                self.original_image = cv2.imread(fname)
                self.image_path = fname
            else:
                QMessageBox.warning(self, "Error", "Could not open or display the image.")
    def convert_to_grayscale(self):
        if hasattr(self, 'original_image'):
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            height, width = grayscale_image.shape
            q_img = QImage(grayscale_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def convert_to_black_and_white(self):
        if hasattr(self, 'original_image'):
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, black_white_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
            height, width = black_white_image.shape
            q_img = QImage(black_white_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def convert_to_complement_image(self):
        if hasattr(self, 'original_image'):
            image = self.original_image.copy()
            height, width = image.shape[:2]
            for x in range(width):
                for y in range(height):
                    pixel = image[y, x]
                    new_pixel = tuple(255 - component for component in pixel)
                    image[y, x] = new_pixel

            q_img = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_13.setPixmap(pixmap)
            self.label_13.setAlignment(Qt.AlignCenter)        
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
#-------------------------group6------------------------------------

    def apply6_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_10()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                if selected_radio == "Salt A Pepper":
        
                    self.apply_salt_and_pepper_noise()
                    
                elif selected_radio == "Uniform":
                    self.add_uniform_noise()   
                    

            else:
                # If no radio button is selected, show a message box
                QMessageBox.warning(self, "No Radio Button Selected", "Please select a radio button!")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def get_selected_radio_in_groupbox_10(self):
        group_box = self.findChild(QGroupBox, "groupBox_10")
        if group_box:
            for child in group_box.findChildren(QRadioButton):
                if child.isChecked():
                    print("Selected radio button:", child.text())  # Check which radio button is selected
                    return child.text()
        else:
            print("groupBox_10 not found")
        return None
    def add_uniform_noise(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the original image to grayscale
                if len(self.original_image.shape) == 3:
                    # If the image has multiple channels, convert to grayscale
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    # Image is already grayscale
                    gray_image = self.original_image.copy()

                # Get image dimensions
                height, width = gray_image.shape

                # Get the start, end, and percentage values from line edits
                noise_start_str = self.lineEdit_7.text()
                noise_end_str = self.lineEdit_8.text()
                noise_percentage_str = self.lineEdit_9.text()

                # Validate and convert noise values
                try:
                    noise_start = float(noise_start_str)
                    noise_end = float(noise_end_str)
                    noise_percentage = float(noise_percentage_str)

                    # Validate the percentage value
                    if not 0 <= noise_percentage <= 1:
                        raise ValueError("Percentage must be in the range [0, 1].")

                    # Validate start and end values
                    if not 0 <= noise_start <= noise_end <= 255:
                        raise ValueError("Invalid noise values. Start must be less than or equal to end, and both must be in the range [0, 255].")

                except ValueError:
                    raise ValueError("Invalid noise values. Please enter valid numbers.")

                # Create a copy of the image to add noise
                noisy_image = gray_image.copy()

                # Determine the number of pixels to add noise to based on percentage
                num_pixels = int(noise_percentage * height * width)

                # Create a meshgrid of coordinates
                x_coords, y_coords = np.meshgrid(*[np.arange(i) for i in (width, height)])

                # Flatten the coordinates
                flat_coords = np.vstack((x_coords.flatten(), y_coords.flatten())).T

                # Ensure num_pixels does not exceed the total number of pixels
                num_pixels = min(num_pixels, len(flat_coords))

                # Randomly shuffle the coordinates and noise values
                indices = np.random.choice(len(flat_coords), num_pixels, replace=False)
                selected_coords = flat_coords[indices]

                # Generate noise values for the selected pixels
                selected_noise = np.random.uniform(noise_start, noise_end, num_pixels)

                # Assign the noise values to the corresponding coordinates
                noisy_image[selected_coords[:, 1].astype(int), selected_coords[:, 0].astype(int)] = selected_noise

                # Convert NumPy array to QImage for display
                q_img = QImage(noisy_image.data, width, height, width, QImage.Format_Grayscale8)

                # Display the image with noise using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error adding noise to image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def apply_salt_and_pepper_noise(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the original image to grayscale
                if len(self.original_image.shape) == 3:
                    # If the image has multiple channels, convert to grayscale
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    # Image is already grayscale
                    gray_image = self.original_image.copy()

                # Get image dimensions
                height, width = gray_image.shape

                # Get the salt and pepper values from line edits
                salt_value_str = self.lineEdit_5.text()
                pepper_value_str = self.lineEdit_6.text()

                # Validate and convert salt and pepper values
                salt_value = float(salt_value_str) if salt_value_str else 0.0
                pepper_value = float(pepper_value_str) if pepper_value_str else 0.0

                # Validate the salt and pepper values
                if not 0 <= salt_value <= 1 or not 0 <= pepper_value <= 1:
                    raise ValueError("Salt and pepper values must be in the range [0, 1].")

                # Check if both salt and pepper values are empty
                if salt_value == 0.0 and pepper_value == 0.0:
                    raise ValueError("Enter a value for salt, pepper, or both.")

                # Create a copy of the image to add noise
                noisy_image = gray_image.copy()

                # Apply salt noise
                if salt_value > 0.0:
                    salt_mask = np.random.rand(height, width) < salt_value
                    noisy_image[salt_mask] = 255

                # Apply pepper noise
                if pepper_value > 0.0:
                    pepper_mask = np.random.rand(height, width) < pepper_value
                    noisy_image[pepper_mask] = 0

                # Convert NumPy array to QImage for display
                q_img = QImage(noisy_image.data, width, height, width, QImage.Format_Grayscale8)

                # Display the image with salt and/or pepper noise using QLabel (modify as needed)
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                self.label_13.setPixmap(pixmap)

            except ValueError as ve:
                error_message = f"Error: {str(ve)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

            except Exception as e:
                error_message = f"Error adding salt and pepper noise to image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.") 

    def apply_fourier_transform(self):
        if hasattr(self, 'original_image'):
            try:
                # Convert the image to grayscale if it's in color
                if len(self.original_image.shape) == 3:
                    image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    image = self.original_image.copy()

                # Perform Fourier Transformation
                f_transform = np.fft.fft2(image)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

                # Normalize the pixel values to the range [0, 255] for display
                magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Convert NumPy array to QImage for display
                height, width = magnitude_spectrum_normalized.shape
                bytes_per_line = width
                q_img = QImage(magnitude_spectrum_normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                # Convert QImage to QPixmap for display on QLabel
                pixmap = QPixmap.fromImage(q_img)
                pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)

                # Display the Fourier Transform on the QLabel
                self.label_13.setPixmap(pixmap)
                self.label_13.setAlignment(Qt.AlignCenter)

            except Exception as e:
                error_message = f"Error applying Fourier Transform: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok | QMessageBox.Copy)
                if self.sender().standardButton(QMessageBox.Copy) == QMessageBox.Copy:
                    clipboard = QApplication.clipboard()
                    clipboard.setText(error_message)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    
    def save_image(self):
        if hasattr(self, 'original_image'):
            # Get the file path from the user
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)")

            if file_path:
                # Convert QImage to QPixmap
                pixmap = self.label_13.pixmap()
                if not pixmap.isNull():
                    # Save the QPixmap to the specified file path
                    pixmap.save(file_path)

                    QMessageBox.information(self, "Save Successful", f"Image saved to:\n{file_path}", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Error", "No image to save.", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")

    def reset_image(self):
        if hasattr(self, 'original_image'):
            try:
                # Display the original image
                original_height, original_width, _ = self.original_image.shape
                original_bytes_per_line = 3 * original_width
                original_q_img = QImage(
                    self.original_image.data, original_width, original_height, original_bytes_per_line, QImage.Format_RGB888
                )

                # Scale the original image to fit the label
                original_pixmap = QPixmap.fromImage(original_q_img)
                original_pixmap = original_pixmap.scaled(
                    self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio
                )
                self.label_13.setPixmap(original_pixmap)

                # Remove the output image
                self.label_13.clear()

                # Reset any additional settings or attributes in the program
                # Add your code here to reset any additional settings or attributes

            except Exception as e:
                error_message = f"Error resetting image: {str(e)}"
                QMessageBox.warning(self, "Error", error_message, QMessageBox.Ok)

        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def add_value_to_image(self):
        if hasattr(self, 'original_image'):
            value_text = self.lineEdit.text()
            if value_text:
                try:
                    value = int(value_text)
                    image = self.original_image.copy()
                    height, width = image.shape[:2]
                    for x in range(width):
                        for y in range(height):
                            pixel = image[y, x]
                            new_pixel = tuple(np.clip(pixel + value, 0, 255))
                            image[y, x] = new_pixel

                    q_img = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                    self.label_13.setPixmap(pixmap)
                    self.label_13.setAlignment(Qt.AlignCenter)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Please enter a valid integer value.")
            else:
                QMessageBox.warning(self, "Error", "Please enter a value before processing.")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def sub_value_from_image(self):
        if hasattr(self, 'original_image'):
            value_text = self.lineEdit.text()
            if value_text:
                try:
                    value = int(value_text)
                    value = -1 * value
                    image = self.original_image.copy()
                    height, width = image.shape[:2]
                    for x in range(width):
                        for y in range(height):
                            pixel = image[y, x]
                            new_pixel = tuple(np.clip(pixel + value, 0, 255))
                            image[y, x] = new_pixel

                    q_img = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                    self.label_13.setPixmap(pixmap)
                    self.label_13.setAlignment(Qt.AlignCenter)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Please enter a valid integer value.")
            else:
                QMessageBox.warning(self, "Error", "Please enter a value before processing.")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def mul_value_to_image(self):
        if hasattr(self, 'original_image'):
            value_text = self.lineEdit.text()
            if value_text:
                try:
                    value = int(value_text)
                    image = self.original_image.copy()
                    height, width = image.shape[:2]
                    for x in range(width):
                        for y in range(height):
                            pixel = image[y, x]
                            new_pixel = tuple(min(255, int(component * value)) for component in pixel)
                            image[y, x] = new_pixel

                    q_img = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                    self.label_13.setPixmap(pixmap)
                    self.label_13.setAlignment(Qt.AlignCenter)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Please enter a valid integer value.")
            else:
                QMessageBox.warning(self, "Error", "Please enter a value before processing.")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    def div_value_from_image(self):
        if hasattr(self, 'original_image'):
            value_text = self.lineEdit.text()
            if value_text:
                try:
                    value = int(value_text)
                    image = self.original_image.copy()
                    height, width = image.shape[:2]
                    for x in range(width):
                        for y in range(height):
                            pixel = image[y, x]
                            new_pixel = tuple(max(0, int(component / value)) for component in pixel)
                            image[y, x] = new_pixel

                    q_img = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    pixmap = pixmap.scaled(self.label_13.width(), self.label_13.height(), aspectRatioMode=Qt.KeepAspectRatio)
                    self.label_13.setPixmap(pixmap)
                    self.label_13.setAlignment(Qt.AlignCenter)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Please enter a valid integer value.")
            else:
                QMessageBox.warning(self, "Error", "Please enter a value before processing.")
        else:
            QMessageBox.warning(self, "Error", "Please open an image first.")
    

def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
