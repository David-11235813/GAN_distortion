import os
import time
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy,
                               QFileDialog, QSlider, QSpinBox, QFormLayout, QFrame)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from utils_backend import LoadModel, GenerateImage


def load_image(picture_frame, file_path) -> QPixmap | None:
    original_pixmap = QPixmap(file_path)
    if original_pixmap.isNull(): return None
    update_image(original_pixmap, picture_frame)
    return original_pixmap


def update_image(pixmap, picture_frame):
    if pixmap is None: return
    picture_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    target_size = picture_frame.contentsRect().size()
    scaled = pixmap.scaled(
        target_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    picture_frame.setPixmap(scaled)

def init_picture_label(minX: int = 400, minY: int = 400, msg: str = "No image loaded") -> QLabel:
    picture_frame = QLabel()
    # self.picture_frame.setFrameStyle(QFrame.Box | QFrame.Plain)
    picture_frame.setAlignment(Qt.AlignCenter)
    picture_frame.setMinimumSize(minX, minY)
    # self.picture_frame.setMaximumSize(600, 600)  # Max dimensions
    picture_frame.setScaledContents(False)
    picture_frame.setText(msg)
    #picture_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return picture_frame


#todo: load settings panel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Generator Application")
        self.setGeometry(100, 100, 900, 700)

        # Initialize displays
        self.display1 = None
        self.display2 = None
        self.display3 = None

        self.show_display1()

        # Set Display1 as initial central widget
        #self.setCentralWidget(self.display1)

    def show_display1(self):
        #self.setCentralWidget(self.display1)
        self.display1 = Display1(self)
        self.setCentralWidget(self.display1)

    def show_display2(self):
        # BACKEND: Get generated image path from generate_image() call
        # generated_image_path = result of generate_image()

        params = {
            'param1': self.display1.slider1.value(),
            'param2': self.display1.slider2.value()
        }

        self.display2 = Display2(
            self,
            self.display1.current_image_path,
            self.display1.current_image_path,  # BACKEND: Pass generated_image_path here
            params
        )
        self.setCentralWidget(self.display2)

    def show_display3(self, bundle_path):
        # TODO: Create and show Display3
        print(f"Transitioning to Display3 with bundle: {bundle_path}")
        pass


# BACKEND INTEGRATION POINTS:
# - generate_image() function should be imported from your backend module
# - load_image_bundle() function for Button2 functionality

class Display1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = None
        self.original_pixmap = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left side - main controls and image
        left_layout = QVBoxLayout()

        # button_select (pick an image) + button_X (reset)
        top_buttons_layout = QHBoxLayout()

        button1_layout = QHBoxLayout()
        self.button_select = QPushButton("Select image")
        self.button_select.clicked.connect(self.button_select_image)

        self.button_X = QPushButton("X")
        self.button_X.setMaximumWidth(30)
        self.button_X.clicked.connect(self.button_reset_state)
        button1_layout.addWidget(self.button_select)
        button1_layout.addWidget(self.button_X)

        button1_layout.setContentsMargins(0, 0, 50, 0)

        # button_read (read an already generated image bundle)
        self.button_read = QPushButton("OR: Read an already generated image bundle")
        self.button_read.clicked.connect(self.button_read_image_bundle)

        top_buttons_layout.addLayout(button1_layout)
        top_buttons_layout.addWidget(self.button_read)
        left_layout.addLayout(top_buttons_layout)


        # Picture frame
        self.picture_frame = init_picture_label(minX=400, minY=400, msg="No image loaded")
        left_layout.addWidget(self.picture_frame)

        main_layout.addLayout(left_layout, 2)

        #---------------------------------------------------------------------------------------------

        # Right side - side panel with options
        right_layout = QVBoxLayout()
        side_panel = QWidget()
        side_panel.setMaximumWidth(250)
        side_panel_layout = QFormLayout()

        # Example sliders and numeric inputs (customize as needed)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(50)
        self.spin1 = QSpinBox()
        self.spin1.setRange(0, 100)
        self.spin1.setValue(50)
        self.slider1.valueChanged.connect(self.spin1.setValue)
        self.spin1.valueChanged.connect(self.slider1.setValue)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(50)
        self.spin2 = QSpinBox()
        self.spin2.setRange(0, 100)
        self.spin2.setValue(50)
        self.slider2.valueChanged.connect(self.spin2.setValue)
        self.spin2.valueChanged.connect(self.slider2.setValue)

        side_panel_layout.addRow("Parameter 1:", self.slider1)
        side_panel_layout.addRow("", self.spin1)
        side_panel_layout.addRow("Parameter 2:", self.slider2)
        side_panel_layout.addRow("", self.spin2)

        side_panel.setLayout(side_panel_layout)
        right_layout.addWidget(side_panel)


        # button_generate (Generate image)
        self.button_generate = QPushButton("Generate image")
        self.button_generate.setEnabled(False)
        self.button_generate.clicked.connect(self.button_generate_image)
        right_layout.addWidget(self.button_generate)

        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        update_image(self.original_pixmap, self.picture_frame)

    #todo
    def button_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image",
            "generated",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            self.current_image_path = file_path
            self.original_pixmap = load_image(self.picture_frame, file_path)
            self.button_select.setText(Path(file_path).name + f" [{self.original_pixmap.width()}px x {self.original_pixmap.height()}px]")
            self.button_read.setEnabled(False)
            self.button_read.setStyleSheet("color: gray;")
            self.button_generate.setEnabled(True)
    #todo
    def button_reset_state(self):
        self.current_image_path = None
        self.original_pixmap = None
        self.picture_frame.clear()
        self.picture_frame.setAlignment(Qt.AlignCenter)
        self.picture_frame.setText("No image loaded")

        self.button_select.setText("Select image")
        self.button_read.setEnabled(True)
        self.button_read.setStyleSheet("")
        self.button_generate.setEnabled(False)
    #todo
    def button_read_image_bundle(self):
        # BACKEND: Adjust directory and file filter as needed
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Read Generated Image Bundle",
            "",  # Specify default directory here if needed
            "Bundle Files (*.bundle);;All Files (*.*)"  # Adjust extension
        )

        if file_path:
            # BACKEND: Call your load_image_bundle(file_path) function here
            # Then transition to Display3
            self.parent().show_display3(file_path)
    #todo
    def button_generate_image(self):
        # BACKEND: Call your generate_image() function here
        # Pass parameters from sliders/spinboxes as needed:
        # param1 = self.slider1.value()
        # param2 = self.slider2.value()
        # generate_image(self.current_image_path, param1, param2)

        # Transition to Display2
        self.parent().show_display2()




class Display2(QWidget):
    #todo
    def __init__(self, parent=None, original_image_path=None, generated_image_path=None, params=None):
        super().__init__(parent)
        self.original_image_path = original_image_path
        self.generated_image_path = generated_image_path
        self.params = params or {}
        self.left_panel_visible = True
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left-center space container
        images_container = QHBoxLayout()

        # Left panel (toggleable) - original image
        self.left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        self.original_frame = init_picture_label(minX=200, minY=400, msg="Original Image")
        left_panel_layout.addWidget(self.original_frame)
        self.left_panel.setLayout(left_panel_layout)
        images_container.addWidget(self.left_panel, 1)
        self.original_pixmap = load_image(self.original_frame, self.original_image_path)


        # Central panel - generated image
        central_panel = QWidget()
        central_panel_layout = QVBoxLayout()
        self.generated_frame = init_picture_label(minX=200, minY=400, msg="Generated Image")
        central_panel_layout.addWidget(self.generated_frame)
        central_panel.setLayout(central_panel_layout)
        images_container.addWidget(central_panel, 1)
        self.generated_pixmap = load_image(self.generated_frame, self.generated_image_path)


        main_layout.addLayout(images_container, 3)

        #---------------------------------------------------------------------------------------------

        # Right panel - options
        right_layout = QVBoxLayout()
        right_panel = QWidget()
        right_panel.setMaximumWidth(250)
        right_panel_layout = QFormLayout()

        # Example sliders and numeric inputs (should match Display1's side panel)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(self.params.get('param1', 50))
        self.spin1 = QSpinBox()
        self.spin1.setRange(0, 100)
        self.spin1.setValue(self.params.get('param1', 50))
        self.slider1.valueChanged.connect(self.spin1.setValue)
        self.spin1.valueChanged.connect(self.slider1.setValue)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(self.params.get('param2', 50))
        self.spin2 = QSpinBox()
        self.spin2.setRange(0, 100)
        self.spin2.setValue(self.params.get('param2', 50))
        self.slider2.valueChanged.connect(self.spin2.setValue)
        self.spin2.valueChanged.connect(self.slider2.setValue)

        right_panel_layout.addRow("Parameter 1:", self.slider1)
        right_panel_layout.addRow("", self.spin1)
        right_panel_layout.addRow("Parameter 2:", self.slider2)
        right_panel_layout.addRow("", self.spin2)

        right_panel.setLayout(right_panel_layout)
        right_layout.addWidget(right_panel)


        # Buttons below right panel
        buttons_layout = QVBoxLayout()
        self.button_back = QPushButton("Back")
        self.button_back.clicked.connect(self.button_go_back)
        self.button_toggle = QPushButton("Disable og image")
        self.button_toggle.clicked.connect(self.button_toggle_original)
        self.button_regenerate = QPushButton("Regenerate")
        self.button_regenerate.clicked.connect(self.button_regenerate_image)
        self.button_save = QPushButton("Save bundle")
        self.button_save.clicked.connect(self.button_save_bundle)

        buttons_layout.addWidget(self.button_back)
        buttons_layout.addWidget(self.button_toggle)
        buttons_layout.addWidget(self.button_regenerate)
        buttons_layout.addWidget(self.button_save)

        right_layout.addLayout(buttons_layout)

        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

    def showEvent(self, event):
        """Called when the widget is shown for the first time; ensure images are updated."""
        super().showEvent(event)
        # Ensure we attempt an initial update after layout/show
        update_image(self.original_pixmap, self.original_frame)
        update_image(self.generated_pixmap, self.generated_frame)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        update_image(self.original_pixmap, self.original_frame)
        update_image(self.generated_pixmap, self.generated_frame)

    def button_toggle_original(self):
        self.left_panel_visible = not self.left_panel_visible
        self.left_panel.setVisible(self.left_panel_visible)

        if self.left_panel_visible: self.button_toggle.setText("Disable og image")
        else: self.button_toggle.setText("Enable og image")

    def button_go_back(self):
        self.parent().show_display1()

    #todo
    def button_regenerate_image(self):
        # BACKEND: Call generate_image() again with current parameters
        # param1 = self.slider1.value()
        # param2 = self.slider2.value()
        # new_image_path = generate_image(self.original_image_path, param1, param2)
        # self.load_image(self.generated_frame, new_image_path)
        pass
    #todo
    def button_save_bundle(self):
        # BACKEND: Save bundle with original image, generated image, and parameters
        # bundle_data = {
        #     'original': self.original_image_path,
        #     'generated': self.generated_image_path,
        #     'params': {'param1': self.slider1.value(), 'param2': self.slider2.value()}
        # }

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Bundle",
            "",
            "Bundle Files (*.bundle);;All Files (*.*)"
        )

        if file_path:
            # BACKEND: Call your save_bundle(file_path, bundle_data) function here
            pass


#previous
class MainWindowPREV(QWidget):
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
