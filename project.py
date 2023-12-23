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
        self.pushButton_17.clicked.connect(self.apply6_button_clicked)
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

def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
