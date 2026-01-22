import os
import time
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from utils_backend import LoadModel, GenerateImage


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GAN Image Generator')
        self.resize(400, 400)

        # Load model
        self.G = LoadModel()

        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'generated')
        os.makedirs(self.output_dir, exist_ok=True)


        # UI elements
        self.img_label = QLabel('Click "Generate" to create an image')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(280, 280)

        self.gen_btn = QPushButton('Generate Image')
        self.gen_btn.clicked.connect(self.generate_image)

        self.save_btn = QPushButton('Save Last Image')
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.img_label)
        layout.addWidget(self.gen_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

        self.last_img_path = None


    def generate_image(self):

        # Generate and save temporarily
        fake_img, self.last_img_path = GenerateImage(self.G, self.output_dir)

        # Display
        pixmap = QPixmap(self.last_img_path).scaled(280, 280, Qt.KeepAspectRatio)
        self.img_label.setPixmap(pixmap)
        self.save_btn.setEnabled(True)


    def save_image(self):
        if self.last_img_path:
            img_name = f"generated_{time.strftime('%Y%m%d-%H%M%S')}.png"
            save_path = os.path.join(self.output_dir, img_name)

            pixmap = QPixmap(self.last_img_path)
            pixmap.save(save_path)

            self.setWindowTitle(f'Saved: {img_name}')
