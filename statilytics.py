import importlib.util, os
from PySide6 import QtWidgets
def _import_mod():
    here=os.path.dirname(__file__)
    path=os.path.join(os.path.dirname(here),'vendor','statilytics_studio_v1_0.py')
    spec=importlib.util.spec_from_file_location('statilytics_studio_v1_0',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def open_statilytics():
    mod=_import_mod()
    MW=getattr(mod,'MainWindow',None)
    if MW is None:
        QtWidgets.QMessageBox.critical(None,'Error','Statilytics missing MainWindow'); return None
    try: w=MW()
    except TypeError: w=MW('Guest')
    w.show(); return w
