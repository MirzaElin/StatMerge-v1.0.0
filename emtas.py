import importlib.util, os, types, sys
from PySide6 import QtWidgets
def _ensure_metrics():
    if 'metrics' in sys.modules: return
    from ..metrics_mod import agreement as _ag
    m=types.ModuleType('metrics')
    m._wilson_ci=_ag._wilson_ci; m._safe_div=_ag._safe_div; m.binary_metrics=_ag.binary_metrics
    m.roc_points_auc=_ag.roc_points_auc; m.pr_points_ap=_ag.pr_points_ap
    m.fleiss_kappa_from_raw=_ag.fleiss_kappa_from_raw; m.icc2_1=_ag.icc2_1; m.bootstrap_icc_ci=_ag.bootstrap_icc_ci
    sys.modules['metrics']=m
def _import_mod():
    here=os.path.dirname(__file__)
    path=os.path.join(os.path.dirname(here),'vendor','EMTAS_v1_0.py')
    spec=importlib.util.spec_from_file_location('EMTAS_v1_0',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def open_emtas():
    _ensure_metrics(); mod=_import_mod()
    if hasattr(mod,'LoginDialog'):
        try: mod.LoginDialog.exec=lambda self: QtWidgets.QDialog.Accepted; mod.LoginDialog.exec_=lambda self: QtWidgets.QDialog.Accepted
        except Exception: pass
    MW=getattr(mod,'MainWindow',None)
    if MW is None:
        QtWidgets.QMessageBox.critical(None,'Error','EMTAS missing MainWindow'); return None
    try: w=MW('Guest')
    except TypeError: w=MW()
    w.show(); return w
