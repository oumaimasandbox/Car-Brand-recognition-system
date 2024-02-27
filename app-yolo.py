from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsView, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,QGraphicsScene,QFileDialog, QLineEdit,
    QWidget)
import sys,cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
from PySide6.QtWidgets import QMessageBox

from torchvision import transforms



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(356, 653)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.view = QGraphicsView(self.centralwidget)
        self.view.setObjectName(u"view")
        self.view.setGeometry(QRect(50, 70, 256, 271))
        self.importButton = QPushButton(self.centralwidget)
        self.importButton.setObjectName(u"importButton")
        self.importButton.setGeometry(QRect(100, 380, 151, 24))
        self.importButton.setStyleSheet(u"border: 2px solid white ;\n"
"color: white;\n"
"margin:2px;\n"
"")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(-10, -30, 371, 671))
        self.label.setStyleSheet(u"border-image: url(/Users/oumaima/Downloads/S7/Pattern recognition/bg.jpeg) 1 1 1 1 stretch stretch;\n"
"")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(30, 550, 161, 41))
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"font: 700 italic 15pt \"Segoe UI\";")
        self.predictButton = QPushButton(self.centralwidget)
        self.predictButton.setObjectName(u"predictButton")
        self.predictButton.setGeometry(QRect(100, 470, 151, 24))
        self.predictButton.setStyleSheet(u"border: 2px solid white;\n"
"color: white;margin:2px;")
        self.resultat = QLineEdit(self.centralwidget)
        self.resultat.setObjectName(u"resultat")
        self.resultat.setGeometry(QRect(180, 560, 131, 22))
        MainWindow.setCentralWidget(self.centralwidget)
        self.label.raise_()
        self.view.raise_()
        self.importButton.raise_()
        self.label_2.raise_()
        self.predictButton.raise_()
        self.resultat.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 356, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.importButton.setText(QCoreApplication.translate("MainWindow", u"Importer", None))
        self.label.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" color:#ffffff;\">Predicted lo  </span></p></body></html>", None))
        self.predictButton.setText(QCoreApplication.translate("MainWindow", u"Predict", None))
    # retranslateUi
        
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.current_image = None  
        self.importButton.clicked.connect(self.importImage)
        self.model = self.load_model('/Users/oumaima/Downloads/best.pt')
        self.predictButton.clicked.connect(self.predictImage)

    def importImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)

        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Resize the image to fit within the QGraphicsView
                pixmap = QPixmap.fromImage(q_image)
                pixmap = pixmap.scaled(self.view.width(), self.view.height(), Qt.KeepAspectRatio)

                scene = QGraphicsScene()
                scene.addPixmap(pixmap)
                self.view.setScene(scene)

                # Store the current image
                self.current_image = image_rgb

    def load_model(self, model_path):
        model = YOLO(model_path)
        return model
    import os

    def predictImage(self):
        if self.current_image is not None:
            with torch.no_grad():
                # Run the prediction
                results = self.model.predict(self.current_image,save=True)

            # Check if there are any detections
            if results and results[0].boxes:
                first_result = results[0]
                first_box = first_result.boxes[0]

                # Extract the class index and get the class name
                class_index = int(first_box.cls[0].item())
                predicted_class = first_result.names[class_index]

                # Set the predicted class name in the QLineEdit widget
                self.resultat.setText(predicted_class)
            else:
                # No detections, show a popup message
                self.resultat.setText("")
                msg = QMessageBox()
                msg.setWindowTitle("Prediction Result")
                msg.setText("Can't predict the image, you may need to give another picture.")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

            # Base directory where YOLO saves output images
            base_output_dir = '/Users/oumaima/runs/detect'

            # Find the most recent 'predict' folder
            latest_predict_folder = max([os.path.join(base_output_dir, d) for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith('predict')], key=os.path.getmtime)

            # Find the latest file in the most recent 'predict' folder
            latest_file = max([os.path.join(latest_predict_folder, f) for f in os.listdir(latest_predict_folder) if os.path.isfile(os.path.join(latest_predict_folder, f))], key=os.path.getctime)

            # Load the output image
            output_image = cv2.imread(latest_file)
            if output_image is not None:
                # Convert BGR to RGB
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

                height, width, channel = output_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(output_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Resize the image to fit within the QGraphicsView
                pixmap = QPixmap.fromImage(q_image)
                pixmap = pixmap.scaled(self.view.width(), self.view.height(), Qt.KeepAspectRatio)

                scene = QGraphicsScene()
                scene.addPixmap(pixmap)
                self.view.setScene(scene)
                

        





   

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec())