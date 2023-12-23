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

        
        
        
        
        
        
        
        
        
        
        
        
        #group 3
        self.pushButton_14.clicked.connect(self.apply3_button_clicked)

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
    
#----------------- point processing for edge detection------------------
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

        
def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()