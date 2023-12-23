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

        
def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()