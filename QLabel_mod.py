from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class QLabel_click(QLabel):
    clicked = pyqtSignal()

    def __init(self, parent):
        QLabel.__init__(self, QMouseEvent)

    def mousePressEvent(self, ev):
        self.clicked.emit()
