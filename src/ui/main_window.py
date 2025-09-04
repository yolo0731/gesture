from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic
from PyQt5.QtGui import QFont
from utils import paths


class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # load .ui from resources folder
        self.ui = uic.loadUi(str(paths.ui_dir() / "interaction.ui"))
        self.ui.textBrowser.setFont(QFont('华文楷体', 20))

        # Wire buttons
        self.ui.rec_again.clicked.connect(self.clear_log)
        self.ui.number_rec.clicked.connect(self.number_rec)
        self.ui.letter_rec.clicked.connect(self.letter_rec)
        self.ui.number_gesture.clicked.connect(self.number_gesture)
        self.ui.letter_gesture.clicked.connect(self.letter_gesture)
        self.ui.number_rec_2.clicked.connect(self.dynamic_rec)
        if hasattr(self.ui, 'direction_gesture'):
            self.ui.direction_gesture.clicked.connect(self.direction_gesture)
        if hasattr(self.ui, 'letter_gesture_dynamic'):
            self.ui.letter_gesture_dynamic.clicked.connect(self.dynamic_letter_gesture)

    def clear_log(self):
        self.ui.textBrowser.clear()

    def number_rec(self):
        from recognition.number_rec import number_rec
        number_rec(self.ui)

    def letter_rec(self):
        from recognition.letter_rec import letter_rec
        letter_rec(self.ui)

    def number_gesture(self):
        from recognition.number_gesture import number_gesture
        number_gesture(self.ui)

    def letter_gesture(self):
        from recognition.letter_gesture import letter_gesture
        letter_gesture(self.ui)

    def dynamic_letter_gesture(self):
        from recognition.letter_gesture import JZRec
        JZRec(self.ui)

    def dynamic_rec(self):
        from ml.dg_prediction_CSRN import CSRN
        CSRN(self.ui)

    def direction_gesture(self):
        from recognition.direction_gesture import direction_gesture
        direction_gesture(self.ui)
