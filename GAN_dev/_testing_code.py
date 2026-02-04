import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QHBoxLayout
)
from PySide6.QtWidgets import QFileDialog

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


IMAGE_SIZE = 512


def scale_pixmap(pixmap: QPixmap) -> QPixmap:
    return pixmap.scaled(
        IMAGE_SIZE,
        IMAGE_SIZE,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )


def to_grayscale(image: QImage) -> QImage:
    return image.convertToFormat(QImage.Format_Grayscale8)


def invert_colors(image: QImage) -> QImage:
    img = image.copy()
    img.invertPixels()
    return img


class ImageWindow(QWidget):
    def __init__(self, image_path):
        super().__init__()

        self.setWindowTitle("Minimal Image Viewer")
        self.setMinimumSize(IMAGE_SIZE * 3, IMAGE_SIZE)

        layout = QHBoxLayout(self)
        layout.setSpacing(10)

        original_pixmap = QPixmap(image_path)
        if original_pixmap.isNull():
            raise RuntimeError("Failed to load image.")

        # ---- Original (scaled to label size) ----
        original_label = QLabel()
        original_label.setFixedSize(IMAGE_SIZE, IMAGE_SIZE)
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setPixmap(scale_pixmap(original_pixmap))

        # ---- Grayscale version ----
        gray_image = to_grayscale(original_pixmap.toImage())
        gray_pixmap = QPixmap.fromImage(gray_image)

        gray_label = QLabel()
        gray_label.setFixedSize(IMAGE_SIZE, IMAGE_SIZE)
        gray_label.setAlignment(Qt.AlignCenter)
        gray_label.setPixmap(scale_pixmap(gray_pixmap))

        # ---- Inverted color version ----
        inverted_image = invert_colors(original_pixmap.toImage())
        inverted_pixmap = QPixmap.fromImage(inverted_image)

        inverted_label = QLabel()
        inverted_label.setFixedSize(IMAGE_SIZE, IMAGE_SIZE)
        inverted_label.setAlignment(Qt.AlignCenter)
        inverted_label.setPixmap(scale_pixmap(inverted_pixmap))

        # Add labels to layout
        layout.addWidget(original_label)
        layout.addWidget(gray_label)
        layout.addWidget(inverted_label)


def select_image_file(parent=None):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Image File",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
    )
    if file_path: return file_path
    return None


def testing():
    app = QApplication(sys.argv)

    # Change this path to your image file
    print("choose image file, the cat isn't available.")
    return

    #IMAGE_PATH = "hi.jpg"
    IMAGE_PATH = None
    #IMAGE_PATH = select_image_file()

    window = ImageWindow(IMAGE_PATH)
    window.show()

    sys.exit(app.exec())
