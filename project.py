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
        self.pushButton_12.clicked.connect(self.apply_button_clicked)
        #group buttons 1
        self.pushButton_12.clicked.connect(self.apply_button_clicked)

        #----------------------- group 1------------------------------------------
        def apply_button_clicked(self):
            if hasattr(self, 'original_image'):
                selected_radio = self.get_selected_radio_in_groupbox_5()
                print ("selected_radio in apply_button_clicked is ",selected_radio)
            
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

def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()