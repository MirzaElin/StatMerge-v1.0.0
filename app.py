import sys
from PySide6 import QtWidgets, QtCore, QtGui
from .datalab import DataLab
from ..integrations.statilytics import open_statilytics
from ..integrations.emtas import open_emtas
APP_TITLE='StatMerge v1.0 — © 2025 Mirza Niaz Zaman Elin'
def _ensure_qaction():
    try: QtWidgets.QAction=QtGui.QAction
    except Exception: pass
def run():
    _ensure_qaction()
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    mw=QtWidgets.QMainWindow(); mw.setWindowTitle(APP_TITLE); mw.resize(1100,720)
    c=QtWidgets.QWidget(mw); mw.setCentralWidget(c); v=QtWidgets.QVBoxLayout(c)
    title=QtWidgets.QLabel('<h1>StatMerge v1.0</h1><p>Unified desktop suite.</p>'); title.setTextFormat(QtCore.Qt.RichText); v.addWidget(title)
    grid=QtWidgets.QGridLayout(); v.addLayout(grid)
    b1=QtWidgets.QPushButton('Open Statilytics Studio'); b2=QtWidgets.QPushButton('Open EMTAS'); b3=QtWidgets.QPushButton('Open Data Lab'); b4=QtWidgets.QPushButton('About / License')
    for b in (b1,b2,b3,b4): b.setMinimumHeight(44)
    grid.addWidget(b1,0,0); grid.addWidget(b2,0,1); grid.addWidget(b3,1,0); grid.addWidget(b4,1,1)
    grid.setColumnStretch(0,1); grid.setColumnStretch(1,1)
    wins=[]
    def _open_stat():
        w=open_statilytics();
        if w: wins.append(w)
    def _open_emtas():
        w=open_emtas();
        if w: wins.append(w)
    def _open_datalab():
        w=DataLab(); w.show(); wins.append(w)
    def _about():
        QtWidgets.QMessageBox.information(mw,'About / License','StatMerge v1.0\nCopyright © 2025 Mirza Niaz Zaman Elin\nLicense: MIT')
    b1.clicked.connect(_open_stat); b2.clicked.connect(_open_emtas); b3.clicked.connect(_open_datalab); b4.clicked.connect(_about)
    mw.show(); run=getattr(app,'exec',None) or getattr(app,'exec_',None); return run()
