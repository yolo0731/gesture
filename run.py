import sys
from pathlib import Path

# Ensure src/ is on sys.path for src-layout
_here = Path(__file__).resolve().parent
_src = _here / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from PyQt5.QtWidgets import QApplication
from ui.main_window import mainwindow


def main():
    app = QApplication(sys.argv)
    window = mainwindow()
    window.ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
