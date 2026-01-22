import sys
from utils_UI import *
from PySide6.QtWidgets import QApplication



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())