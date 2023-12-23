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
        self.pushButton_17.clicked.connect(self.apply6_button_clicked)
#-------------------------group6------------------------------------
    def apply6_button_clicked(self):
        if hasattr(self, 'original_image'):
            selected_radio = self.get_selected_radio_in_groupbox_10()
            print ("selected_radio in apply_button_clicked is ",selected_radio)
            if selected_radio:
                # Perform actions based on the selected radio button
                # if selected_radio == "Salt A Pepper":
        
                #     self.apply_salt_and_pepper_noise()
                    
                if selected_radio == "Uniform":
                 
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
        
def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    