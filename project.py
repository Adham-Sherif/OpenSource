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

        #group 2 button 
        self.pushButton_13.clicked.connect(self.apply2_button_clicked)
        #group buttons 1
        self.pushButton_12.clicked.connect(self.apply_button_clicked)

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
def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
