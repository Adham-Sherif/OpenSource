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

        
def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()