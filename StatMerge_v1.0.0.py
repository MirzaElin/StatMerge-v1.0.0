import sys
import os
import types
import importlib.util
import base64
import math
import traceback
APP_TITLE='StatMerge v1.0 — © 2025 Mirza Niaz Zaman Elin'
from PySide6 import QtWidgets, QtCore, QtGui
try:
    from matplotlib.backends import backend_qtagg as _bqtagg
    sys.modules['matplotlib.backends.backend_qt5agg']=_bqtagg
except Exception:
    pass
try:
    QtWidgets.QAction=QtGui.QAction
except Exception:
    pass
def _shim_pyqt5():
    if 'PyQt5' in sys.modules:
        return
    m=types.ModuleType('PyQt5')
    mw=types.ModuleType('PyQt5.QtWidgets')
    mc=types.ModuleType('PyQt5.QtCore')
    mg=types.ModuleType('PyQt5.QtGui')
    for n in dir(QtWidgets):
        setattr(mw,n,getattr(QtWidgets,n))
    for n in dir(QtCore):
        setattr(mc,n,getattr(QtCore,n))
    for n in dir(QtGui):
        setattr(mg,n,getattr(QtGui,n))
    try:
        mw.QAction=QtGui.QAction
    except Exception:
        pass
    try:
        mc.pyqtSignal=QtCore.Signal
        mc.pyqtSlot=QtCore.Slot
    except Exception:
        pass
    m.QtWidgets=mw
    m.QtCore=mc
    m.QtGui=mg
    sys.modules['PyQt5']=m
    sys.modules['PyQt5.QtWidgets']=mw
    sys.modules['PyQt5.QtCore']=mc
    sys.modules['PyQt5.QtGui']=mg
_shim_pyqt5()
if 'metrics' not in sys.modules:
    mm=types.ModuleType('metrics')
    try:
        import numpy as _np
    except Exception:
        _np=None
    try:
        from sklearn.metrics import roc_curve as _roc_curve
        from sklearn.metrics import auc as _auc
        from sklearn.metrics import precision_recall_curve as _pr_curve
        from sklearn.metrics import average_precision_score as _ap
    except Exception:
        _roc_curve=None
        _auc=None
        _pr_curve=None
        _ap=None
    def _wilson_ci(s,t,z=1.96):
        if not t or s is None:
            return float('nan'),float('nan')
        p=float(s)/float(t)
        d=1.0+(z*z)/t
        c=p+(z*z)/(2*t)
        a=z*math.sqrt((p*(1-p)+(z*z)/(4*t))/t)
        lo=(c-a)/d
        hi=(c+a)/d
        if lo<0:
            lo=0.0
        if lo>1:
            lo=1.0
        if hi<0:
            hi=0.0
        if hi>1:
            hi=1.0
        return float(lo),float(hi)
    def _safe_div(a,b):
        if b:
            return float(a)/float(b)
        return float('nan')
    def binary_metrics(y_true,y_pred):
        tp=0
        fp=0
        tn=0
        fn=0
        for t,p in zip(y_true,y_pred):
            if t==1 and p==1:
                tp+=1
            elif t==0 and p==1:
                fp+=1
            elif t==0 and p==0:
                tn+=1
            elif t==1 and p==0:
                fn+=1
        N=tp+fp+tn+fn
        acc=_safe_div(tp+tn,N)
        sens=_safe_div(tp,tp+fn)
        spec=_safe_div(tn,tn+fp)
        ppv=_safe_div(tp,tp+fp)
        npv=_safe_div(tn,tn+fn)
        f1=_safe_div(2*tp,2*tp+fp+fn)
        if not math.isnan(sens) and not math.isnan(spec):
            bal=0.5*(sens+spec)
            yj=sens+spec-1.0
        else:
            bal=float('nan')
            yj=float('nan')
        den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        if den:
            den=math.sqrt(den)
            mcc=_safe_div(tp*tn-fp*fn,den)
        else:
            mcc=float('nan')
        return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'N':N,'Accuracy':acc,'Accuracy_CI':_wilson_ci(tp+tn,N),'Sensitivity_TPR':sens,'Sensitivity_CI':_wilson_ci(tp,tp+fn),'Specificity_TNR':spec,'Specificity_CI':_wilson_ci(tn,tn+fp),'PPV':ppv,'PPV_CI':_wilson_ci(tp,tp+fp),'NPV':npv,'NPV_CI':_wilson_ci(tn,tn+fn),'F1':f1,'Balanced_Accuracy':bal,'Youdens_J':yj,'MCC':mcc}
    def roc_points_auc(y_true,scores):
        if _roc_curve is None or _auc is None:
            raise ImportError('scikit-learn not available for ROC')
        fpr,tpr,_=_roc_curve(y_true,scores)
        return fpr.tolist(),tpr.tolist(),float(_auc(fpr,tpr))
    def pr_points_ap(y_true,scores):
        if _pr_curve is None or _ap is None:
            raise ImportError('scikit-learn not available for PR')
        precision,recall,_=_pr_curve(y_true,scores)
        ap=_ap(y_true,scores)
        return recall.tolist(),precision.tolist(),float(ap)
    def fleiss_kappa_from_raw(matrix):
        if _np is None:
            raise ImportError('numpy required')
        cats=set()
        for row in matrix:
            for lab in row:
                if lab is None:
                    continue
                s=str(lab).strip()
                if s=='':
                    continue
                cats.add(s)
        cats=sorted(cats)
        if not cats:
            return float('nan'),[],{}
        c2i={c:i for i,c in enumerate(cats)}
        counts=[]
        for row in matrix:
            c=[0.0]*len(cats)
            for lab in row:
                if lab is None:
                    continue
                s=str(lab).strip()
                if s=='':
                    continue
                c[c2i[s]]+=1.0
            counts.append(c)
        n_i=[sum(c) for c in counts]
        valid=[ni>=2 for ni in n_i]
        counts=[c for c,ok in zip(counts,valid) if ok]
        n_i=[ni for ni,ok in zip(n_i,valid) if ok]
        if not counts:
            return float('nan'),[],{}
        P_i=[]
        for c,ni in zip(counts,n_i):
            P_i.append((sum(x*(x-1) for x in c))/(ni*(ni-1)))
        p_j=[0.0]*len(cats)
        for c in counts:
            for j,x in enumerate(c):
                p_j[j]+=x
        den=sum(n_i)
        p_j=[x/den for x in p_j]
        Pbar=sum(P_i)/len(P_i)
        Pe=sum(x*x for x in p_j)
        denom=1-Pe
        if denom!=0:
            kappa=(Pbar-Pe)/denom
        else:
            kappa=float('nan')
        marg={cats[i]:p_j[i] for i in range(len(cats))}
        return float(kappa),P_i,marg
    def bootstrap_fleiss_ci(matrix,B=500,alpha=0.05):
        if _np is None:
            raise ImportError('numpy required')
        import random
        kappas=[]
        for _ in range(B):
            boot=[matrix[random.randint(0,len(matrix)-1)] for __ in range(len(matrix))]
            kv=fleiss_kappa_from_raw(boot)[0]
            if not math.isnan(kv):
                kappas.append(kv)
        if not kappas:
            return float('nan'),float('nan')
        kappas.sort()
        lo=int((alpha/2)*len(kappas))
        hi=int((1-alpha/2)*len(kappas))-1
        if lo<0:
            lo=0
        if lo>=len(kappas):
            lo=len(kappas)-1
        if hi<0:
            hi=0
        if hi>=len(kappas):
            hi=len(kappas)-1
        return float(kappas[lo]),float(kappas[hi])
    def icc2_1(data):
        if _np is None:
            raise ImportError('numpy required')
        import numpy as np
        d = np.asarray(data, dtype=float)
        if np.isnan(d).any():
            d = d[~np.isnan(d).any(axis=1)]
        if d.ndim != 2:
            return float('nan'), {}
        n, k = d.shape
        if n < 2 or k < 2:
            return float('nan'), {}
        ms = d.mean(axis=1, keepdims=True)
        mr = d.mean(axis=0, keepdims=True)
        gm = float(d.mean())
        SSR = float(k * ((ms - gm) ** 2).sum())
        SSC = float(n * ((mr - gm) ** 2).sum())
        SSE = float(((d - ms - mr + gm) ** 2).sum())
        MSR = SSR / max(n - 1, 1)
        MSC = SSC / max(k - 1, 1)
        MSE = SSE / max((n - 1) * (k - 1), 1)
        denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / max(n, 1))
        if not np.isfinite(denom) or denom == 0:
            icc = float('nan')
        else:
            icc = float((MSR - MSE) / denom)
            if np.isfinite(icc):
                icc = max(-1.0, min(1.0, icc))
        comps = {
            'MSR': MSR, 'MSC': MSC, 'MSE': MSE,
            'SSR': SSR, 'SSC': SSC, 'SSE': SSE,
            'n_subjects': int(n), 'k_raters': int(k), 'grand_mean': gm
        }
        return icc, comps
    def bootstrap_icc_distribution(data, B=1000):
        if _np is None:
            raise ImportError('numpy required')
        import numpy as np
        d = np.asarray(data, dtype=float)
        if np.isnan(d).any():
            d = d[~np.isnan(d).any(axis=1)]
        if d.ndim != 2:
            return []
        n, k = d.shape
        if n < 2 or k < 2:
            return []
        vals = []
        for _ in range(int(B)):
            idx = np.random.randint(0, n, size=n)
            try:
                v, _ = icc2_1(d[idx])
                vals.append(float(v) if np.isfinite(v) else np.nan)
            except Exception:
                vals.append(np.nan)
        return [float(x) for x in vals]
    def _nanquantile_safe(vals, qs=(0.025, 0.975)):
        import numpy as np
        arr = np.asarray([float(v) for v in np.asarray(vals).ravel()
                          if v is not None and np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return np.array([np.nan for _ in np.atleast_1d(qs)], dtype=float)
        return np.nanquantile(arr, qs, method='linear')
    def bootstrap_icc_ci(data, B=500, alpha=0.05):
        if _np is None:
            raise ImportError('numpy required')
        vals = bootstrap_icc_distribution(data, B=max(B, 200))
        lo, hi = _nanquantile_safe(vals, (alpha/2.0, 1.0 - alpha/2.0))
        return float(lo), float(hi)
    def icc_avg(icc1, k, reduce='mean'):
        """Spearman–Brown average-measure reliability from single-measure ICCs."""
        import numpy as np
        arr = np.asarray(icc1, dtype=float)
        if k <= 1:
            sb = arr
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                sb = (k * arr) / (1.0 + (k - 1.0) * arr)
        if reduce is None:
            return sb
        if reduce == 'median':
            return float(np.nanmedian(sb))
        return float(np.nanmean(sb))
mm._wilson_ci=_wilson_ci
mm._safe_div=_safe_div
mm.binary_metrics=binary_metrics
mm.roc_points_auc=roc_points_auc
mm.pr_points_ap=pr_points_ap
mm.fleiss_kappa_from_raw=fleiss_kappa_from_raw
mm.bootstrap_fleiss_ci=bootstrap_fleiss_ci
mm.icc2_1=icc2_1
mm.bootstrap_icc_ci=bootstrap_icc_ci
mm.bootstrap_icc_distribution=bootstrap_icc_distribution
mm.icc_avg=icc_avg
sys.modules['metrics']=mm
_STAT_B64='CgppbXBvcnQgc3lzLCBvcywgaW8sIHJlLCBiYXNlNjQsIHRyYWNlYmFjaywgdGV4dHdyYXAsIGNzdiwgd2ViYnJvd3Nlciwgd2ViYnJvd3Nlcgpmcm9tIHR5cGluZyBpbXBvcnQgTGlzdCwgT3B0aW9uYWwsIERpY3QsIEFueQoKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCBodG1sCmltcG9ydCBwYW5kYXMgYXMgcGQKCmZyb20gUHlTaWRlNiBpbXBvcnQgUXRDb3JlLCBRdEd1aSwgUXRXaWRnZXRzCmZyb20gUHlTaWRlNi5RdENvcmUgaW1wb3J0IFF0CmZyb20gUHlTaWRlNi5RdFdpZGdldHMgaW1wb3J0ICgKICAgIFFBcHBsaWNhdGlvbiwgUU1haW5XaW5kb3csIFFGaWxlRGlhbG9nLCBRTWVzc2FnZUJveCwgUVRhYmxlVmlldywKICAgIFFXaWRnZXQsIFFWQm94TGF5b3V0LCBRSEJveExheW91dCwgUUxhYmVsLCBRUHVzaEJ1dHRvbiwgUUdyb3VwQm94LCBRRm9ybUxheW91dCwKICAgIFFDb21ib0JveCwgUVNwaW5Cb3gsIFFMaW5lRWRpdCwgUVRleHRFZGl0LCBRVGFiV2lkZ2V0LCBRQ2hlY2tCb3gsIFFBYnN0cmFjdEl0ZW1WaWV3LAogICAgUVNjcm9sbEFyZWEsIFFTaXplUG9saWN5LCBRVGV4dEJyb3dzZXIsIFFMaXN0V2lkZ2V0LCBRTGlzdFdpZGdldEl0ZW0KKQoKaW1wb3J0IG1hdHBsb3RsaWIKbWF0cGxvdGxpYi51c2UoIkFnZyIpCmltcG9ydCBtYXRwbG90bGliLnB5cGxvdCBhcyBwbHQKaW1wb3J0IGltcG9ydGxpYi5tZXRhZGF0YSBhcyBpbG0KCmZyb20gc2NpcHkgaW1wb3J0IHN0YXRzCmltcG9ydCBzdGF0c21vZGVscy5hcGkgYXMgc20KaW1wb3J0IHN0YXRzbW9kZWxzLmZvcm11bGEuYXBpIGFzIHNtZgpmcm9tIHN0YXRzbW9kZWxzLnN0YXRzLm11bHRpY29tcCBpbXBvcnQgcGFpcndpc2VfdHVrZXloc2QKZnJvbSBzdGF0c21vZGVscy5zdGF0cy5hbm92YSBpbXBvcnQgQW5vdmFSTQpmcm9tIHN0YXRzbW9kZWxzLm11bHRpdmFyaWF0ZS5tYW5vdmEgaW1wb3J0IE1BTk9WQQpmcm9tIHN0YXRzbW9kZWxzLnN0YXRzLmludGVyX3JhdGVyIGltcG9ydCBmbGVpc3Nfa2FwcGEgYXMgc21fZmxlaXNzX2thcHBhCgpmcm9tIHNrbGVhcm4uZGVjb21wb3NpdGlvbiBpbXBvcnQgUENBCmZyb20gc2tsZWFybi5jbHVzdGVyIGltcG9ydCBLTWVhbnMsIEFnZ2xvbWVyYXRpdmVDbHVzdGVyaW5nCmZyb20gc2tsZWFybi5kaXNjcmltaW5hbnRfYW5hbHlzaXMgaW1wb3J0IExpbmVhckRpc2NyaW1pbmFudEFuYWx5c2lzIGFzIExEQQpmcm9tIHNrbGVhcm4ubWV0cmljcyBpbXBvcnQgcm9jX2N1cnZlLCBwcmVjaXNpb25fcmVjYWxsX2N1cnZlLCBjb2hlbl9rYXBwYV9zY29yZSwgc2lsaG91ZXR0ZV9zY29yZQpmcm9tIHNrbGVhcm4ubGluZWFyX21vZGVsIGltcG9ydCBMb2dpc3RpY1JlZ3Jlc3Npb24KZnJvbSBza2xlYXJuLnByZXByb2Nlc3NpbmcgaW1wb3J0IFN0YW5kYXJkU2NhbGVyLCBMYWJlbEVuY29kZXIKZnJvbSBza2xlYXJuLmV4cGVyaW1lbnRhbCBpbXBvcnQgZW5hYmxlX2l0ZXJhdGl2ZV9pbXB1dGVyICAKZnJvbSBza2xlYXJuLmltcHV0ZSBpbXBvcnQgSXRlcmF0aXZlSW1wdXRlcgoKCkhBU19MSUZFTElORVMgPSBUcnVlCnRyeToKICAgIGZyb20gbGlmZWxpbmVzIGltcG9ydCBLYXBsYW5NZWllckZpdHRlciwgQ294UEhGaXR0ZXIsIFdlaWJ1bGxBRlRGaXR0ZXIKZXhjZXB0IEV4Y2VwdGlvbjoKICAgIEhBU19MSUZFTElORVMgPSBGYWxzZQoKSEFTX1BJTkdPVUlOID0gVHJ1ZQp0cnk6CiAgICBpbXBvcnQgcGluZ291aW4gYXMgcGcKZXhjZXB0IEV4Y2VwdGlvbjoKICAgIEhBU19QSU5HT1VJTiA9IEZhbHNlCgpIQVNfRkEgPSBUcnVlCnRyeToKICAgIGZyb20gZmFjdG9yX2FuYWx5emVyIGltcG9ydCBGYWN0b3JBbmFseXplcgpleGNlcHQgRXhjZXB0aW9uOgogICAgSEFTX0ZBID0gRmFsc2UKCkhBU19QUklOQ0UgPSBUcnVlCnRyeToKICAgIGltcG9ydCBwcmluY2UKZXhjZXB0IEV4Y2VwdGlvbjoKICAgIEhBU19QUklOQ0UgPSBGYWxzZQoKSEFTX0tSSVBQID0gVHJ1ZQp0cnk6CiAgICBpbXBvcnQga3JpcHBlbmRvcmZmIGFzIGtkCmV4Y2VwdCBFeGNlcHRpb246CiAgICBIQVNfS1JJUFAgPSBGYWxzZQoKQVBQX05BTUUgPSAiU3RhdGlseXRpY3MgU3R1ZGlvIHYxLjAiCmRlZiBfYXBwX2RpcigpOgogICAgaW1wb3J0IHN5cywgb3MKICAgIHRyeToKICAgICAgICBiYXNlID0gb3MucGF0aC5kaXJuYW1lKG9zLnBhdGguYWJzcGF0aChfX2ZpbGVfXykpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIGJhc2UgPSBvcy5wYXRoLmRpcm5hbWUob3MucGF0aC5hYnNwYXRoKHN5cy5hcmd2WzBdKSkKICAgIGlmIGdldGF0dHIoc3lzLCAiZnJvemVuIiwgRmFsc2UpOgogICAgICAgIGJhc2UgPSBnZXRhdHRyKHN5cywgIl9NRUlQQVNTIiwgYmFzZSkKICAgIHJldHVybiBiYXNlCgpDT1BZUklHSFQgPSAiwqkgMjAyNSBNaXJ6YSBOaWF6IFphbWFuIEVsaW4uIEFsbCByaWdodHMgcmVzZXJ2ZWQuIgpXSU5ET1dfVElUTEUgPSBmIntBUFBfTkFNRX0g4oCUIHtDT1BZUklHSFR9IgoKCmNsYXNzIFBhbmRhc01vZGVsKFF0Q29yZS5RQWJzdHJhY3RUYWJsZU1vZGVsKToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkZj1wZC5EYXRhRnJhbWUoKSwgcGFyZW50PU5vbmUpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18ocGFyZW50KTsgc2VsZi5fZGYgPSBkZgogICAgZGVmIHJvd0NvdW50KHNlbGYsIHBhcmVudD1Ob25lKTogcmV0dXJuIGxlbihzZWxmLl9kZi5pbmRleCkKICAgIGRlZiBjb2x1bW5Db3VudChzZWxmLCBwYXJlbnQ9Tm9uZSk6IHJldHVybiBzZWxmLl9kZi5jb2x1bW5zLnNpemUKICAgIGRlZiBkYXRhKHNlbGYsIGluZGV4LCByb2xlPVF0LkRpc3BsYXlSb2xlKToKICAgICAgICBpZiBub3QgaW5kZXguaXNWYWxpZCgpOiByZXR1cm4gTm9uZQogICAgICAgIGlmIHJvbGUgPT0gUXQuRGlzcGxheVJvbGU6CiAgICAgICAgICAgIHYgPSBzZWxmLl9kZi5pbG9jW2luZGV4LnJvdygpLCBpbmRleC5jb2x1bW4oKV0KICAgICAgICAgICAgcmV0dXJuICIiIGlmIHBkLmlzbmEodikgZWxzZSBzdHIodikKICAgICAgICByZXR1cm4gTm9uZQogICAgZGVmIGhlYWRlckRhdGEoc2VsZiwgc2VjdGlvbiwgb3JpZW50YXRpb24sIHJvbGU9UXQuRGlzcGxheVJvbGUpOgogICAgICAgIGlmIHJvbGUgIT0gUXQuRGlzcGxheVJvbGU6IHJldHVybiBOb25lCiAgICAgICAgcmV0dXJuIHN0cihzZWxmLl9kZi5jb2x1bW5zW3NlY3Rpb25dKSBpZiBvcmllbnRhdGlvbj09UXQuSG9yaXpvbnRhbCBlbHNlIHN0cihzZWxmLl9kZi5pbmRleFtzZWN0aW9uXSkKICAgIGRlZiBzZXREYXRhRnJhbWUoc2VsZiwgZGYpOgogICAgICAgIHNlbGYuYmVnaW5SZXNldE1vZGVsKCk7IHNlbGYuX2RmID0gZGYuY29weSgpOyBzZWxmLmVuZFJlc2V0TW9kZWwoKQoKCmNsYXNzIFJlcG9ydEJ1aWxkZXI6CiAgICBkZWYgX19pbml0X18oc2VsZik6CiAgICAgICAgc2VsZi5wYXJ0cyA9IFtdCiAgICAgICAgc2VsZi5wYXJ0cy5hcHBlbmQodGV4dHdyYXAuZGVkZW50KGYiIiIKICAgICAgICA8aHRtbD48aGVhZD48bWV0YSBjaGFyc2V0PSd1dGYtOCc+CiAgICAgICAgPHN0eWxlPgogICAgICAgIGJvZHkge3sgZm9udC1mYW1pbHk6IC1hcHBsZS1zeXN0ZW0sIFNlZ29lIFVJLCBSb2JvdG8sIEhlbHZldGljYSwgQXJpYWwsIHNhbnMtc2VyaWY7IG1hcmdpbjogMjJweDsgfX0KICAgICAgICAuY2FyZCB7eyBib3JkZXI6IDFweCBzb2xpZCAjZTVlN2ViOyBib3JkZXItcmFkaXVzOiAxMnB4OyBwYWRkaW5nOiAxNHB4OyBtYXJnaW46IDEycHggMDsgfX0KICAgICAgICB0YWJsZSB7eyBib3JkZXItY29sbGFwc2U6IGNvbGxhcHNlOyB3aWR0aDogMTAwJTsgfX0KICAgICAgICB0aCx0ZCB7eyBib3JkZXI6IDFweCBzb2xpZCAjZTVlN2ViOyBwYWRkaW5nOiA2cHg7IHRleHQtYWxpZ246IGxlZnQ7IHdoaXRlLXNwYWNlOiBub3dyYXA7IH19CiAgICAgICAgdGgge3sgYmFja2dyb3VuZDogI2Y5ZmFmYjsgfX0KICAgICAgICBpbWcge3sgbWF4LXdpZHRoOiAxMDAlOyBoZWlnaHQ6IGF1dG87IH19CiAgICAgICAgLm11dGVkIHt7IGNvbG9yOiM2YjcyODA7IH19CiAgICAgICAgPC9zdHlsZT48L2hlYWQ+PGJvZHk+CiAgICAgICAgPGgxPntBUFBfTkFNRX08L2gxPgogICAgICAgIDxwIGNsYXNzPSdtdXRlZCc+e0NPUFlSSUdIVH08L3A+CiAgICAgICAgIiIiKSkKICAgIGRlZiBhZGRfaW5mbyhzZWxmLCB0aXRsZSwgYm9keSk6IHNlbGYucGFydHMuYXBwZW5kKGYiPGRpdiBjbGFzcz0nY2FyZCc+PGgzPnt0aXRsZX08L2gzPjxkaXY+e2JvZHl9PC9kaXY+PC9kaXY+IikKICAgIGRlZiBhZGRfa3Yoc2VsZiwgdGl0bGUsIGt2OiBEaWN0W3N0cixBbnldKToKICAgICAgICByb3dzID0gIiIuam9pbihbZiI8dHI+PHRoIHN0eWxlPSd3aWR0aDoyODBweCc+e2t9PC90aD48dGQ+e3Z9PC90ZD48L3RyPiIgZm9yIGssdiBpbiBrdi5pdGVtcygpXSkKICAgICAgICBzZWxmLnBhcnRzLmFwcGVuZChmIjxkaXYgY2xhc3M9J2NhcmQnPjxoMz57dGl0bGV9PC9oMz48dGFibGU+e3Jvd3N9PC90YWJsZT48L2Rpdj4iKQogICAgZGVmIGFkZF90YWJsZShzZWxmLCBkZjogcGQuRGF0YUZyYW1lLCB0aXRsZT0iVGFibGUiKToKICAgICAgICBpZiBkZiBpcyBOb25lIG9yIGxlbihkZik9PTA6IHNlbGYuYWRkX2luZm8odGl0bGUsICJObyByb3dzLiIpOyByZXR1cm4KICAgICAgICBkZiA9IGRmLmNvcHkoKQogICAgICAgIHRyeToKICAgICAgICAgICAgZGYgPSBkZi5yZXBsYWNlKHtucC5uYW46IiJ9KQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICB0aGVhZCA9ICIiLmpvaW4oW2YiPHRoPntzdHIoYyl9PC90aD4iIGZvciBjIGluIGRmLmNvbHVtbnNdKQogICAgICAgIHJvd3MgPSBbXQogICAgICAgIGZvciBfLHIgaW4gZGYuaXRlcnJvd3MoKTogcm93cy5hcHBlbmQoIjx0cj4iKyIiLmpvaW4oW2YiPHRkPntzdHIodil9PC90ZD4iIGZvciB2IGluIHJdKSsgIjwvdHI+IikKICAgICAgICBzZWxmLnBhcnRzLmFwcGVuZChmIjxkaXYgY2xhc3M9J2NhcmQnPjxoMz57dGl0bGV9PC9oMz48dGFibGU+PHRoZWFkPjx0cj57dGhlYWR9PC90cj48L3RoZWFkPjx0Ym9keT57Jycuam9pbihyb3dzKX08L3Rib2R5PjwvdGFibGU+PC9kaXY+IikKICAgIGRlZiBhZGRfZmlndXJlKHNlbGYsIGZpZywgdGl0bGU9IkZpZ3VyZSIpOgogICAgICAgIGJ1ZiA9IGlvLkJ5dGVzSU8oKTsgZmlnLnNhdmVmaWcoYnVmLCBmb3JtYXQ9J3BuZycsIGRwaT0xNjAsIGJib3hfaW5jaGVzPSd0aWdodCcpOyBwbHQuY2xvc2UoZmlnKQogICAgICAgIGI2NCA9IGJhc2U2NC5iNjRlbmNvZGUoYnVmLmdldHZhbHVlKCkpLmRlY29kZSgnYXNjaWknKQogICAgICAgIHNlbGYucGFydHMuYXBwZW5kKGYiPGRpdiBjbGFzcz0nY2FyZCc+PGgzPnt0aXRsZX08L2gzPjxpbWcgc3JjPSdkYXRhOmltYWdlL3BuZztiYXNlNjQse2I2NH0nLz48L2Rpdj4iKQogICAgZGVmIGh0bWwoc2VsZik6IHJldHVybiAiXG4iLmpvaW4oc2VsZi5wYXJ0cyArIFtmIjxocj48cCBjbGFzcz0nbXV0ZWQnPntBUFBfTkFNRX0g4oCUIHtDT1BZUklHSFR9PC9wPjwvYm9keT48L2h0bWw+Il0pCgoKY2xhc3MgRW5naW5lOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGRmOiBwZC5EYXRhRnJhbWUpOiBzZWxmLmRmID0gZGYuY29weSgpCgogICAgCiAgICBkZWYgZGVzY3JpYmUoc2VsZik6IGQgPSBzZWxmLmRmLmRlc2NyaWJlKGluY2x1ZGU9J2FsbCcpLlQ7IGRbJ21pc3NpbmcnXSA9IHNlbGYuZGYuaXNuYSgpLnN1bSgpOyByZXR1cm4gZAogICAgZGVmIGNyb3NzdGFiKHNlbGYsIGE6IHN0ciwgYjogc3RyLCBmaXNoZXI9RmFsc2UpOgogICAgICAgIHRhYiA9IHBkLmNyb3NzdGFiKHNlbGYuZGZbYV0sIHNlbGYuZGZbYl0pCiAgICAgICAgY2hpMiwgcCwgZG9mLCBfID0gc3RhdHMuY2hpMl9jb250aW5nZW5jeSh0YWIsIGNvcnJlY3Rpb249RmFsc2UpCiAgICAgICAgaW5mbyA9IHsiQ2hpLXNxdWFyZSI6IHJvdW5kKGNoaTIsNCksICJwIjogcm91bmQocCw2KSwgImRmIjogZG9mfQogICAgICAgIGlmIGZpc2hlciBhbmQgdGFiLnNoYXBlPT0oMiwyKTogXywgcGYgPSBzdGF0cy5maXNoZXJfZXhhY3QodGFiKTsgaW5mb1siRmlzaGVyIHAiXSA9IHJvdW5kKHBmLDYpCiAgICAgICAgcmV0dXJuIHRhYiwgaW5mbwoKICAgIAogICAgZGVmIHRfb25lX3NhbXBsZShzZWxmLCBjb2w6IHN0ciwgbXU9MC4wKToKICAgICAgICB4ID0gc2VsZi5kZltjb2xdLmRyb3BuYSgpLmFzdHlwZShmbG9hdCk7IHQsIHAgPSBzdGF0cy50dGVzdF8xc2FtcCh4LCBtdSkKICAgICAgICByZXR1cm4geyJuIjogbGVuKHgpLCAibWVhbiI6IHgubWVhbigpLCAic2QiOiB4LnN0ZChkZG9mPTEpLCAidCI6IHQsICJwIjogcH0KICAgIGRlZiB0X2luZChzZWxmLCBjb2w6IHN0ciwgZ3JvdXA6IHN0ciwgZzEsIGcyLCBlcXVhbF92YXI9RmFsc2UpOgogICAgICAgIHgxID0gc2VsZi5kZi5sb2Nbc2VsZi5kZltncm91cF09PWcxLCBjb2xdLmRyb3BuYSgpLmFzdHlwZShmbG9hdCkKICAgICAgICB4MiA9IHNlbGYuZGYubG9jW3NlbGYuZGZbZ3JvdXBdPT1nMiwgY29sXS5kcm9wbmEoKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgdCxwID0gc3RhdHMudHRlc3RfaW5kKHgxLCB4MiwgZXF1YWxfdmFyPWVxdWFsX3Zhcik7IHJldHVybiB7Im4xIjogbGVuKHgxKSwgIm4yIjogbGVuKHgyKSwgInQiOiB0LCAicCI6IHB9CiAgICBkZWYgdF9wYWlyZWQoc2VsZiwgY29sMTogc3RyLCBjb2wyOiBzdHIpOgogICAgICAgIHgxID0gc2VsZi5kZltjb2wxXS5hc3R5cGUoZmxvYXQpOyB4MiA9IHNlbGYuZGZbY29sMl0uYXN0eXBlKGZsb2F0KQogICAgICAgIG0gPSB+KHgxLmlzbmEoKSB8IHgyLmlzbmEoKSk7IHgxPXgxW21dOyB4Mj14MlttXTsgdCxwID0gc3RhdHMudHRlc3RfcmVsKHgxLCB4Mik7IHJldHVybiB7Im4iOiBsZW4oeDEpLCAidCI6IHQsICJwIjogcH0KICAgIGRlZiBhbm92YV9vbmV3YXkoc2VsZiwgZHY6IHN0ciwgZ3JvdXA6IHN0cik6CiAgICAgICAgZ3JvdXBzID0gW2cuZHJvcG5hKCkuYXN0eXBlKGZsb2F0KS52YWx1ZXMgZm9yIF8sZyBpbiBzZWxmLmRmW1tkdixncm91cF1dLmRyb3BuYSgpLmdyb3VwYnkoZ3JvdXApW2R2XV0KICAgICAgICBGLHAgPSBzdGF0cy5mX29uZXdheSgqZ3JvdXBzKTsgcmV0dXJuIEYscAogICAgZGVmIHR1a2V5X2hzZChzZWxmLCBkdjogc3RyLCBncm91cDogc3RyKToKICAgICAgICBkID0gc2VsZi5kZltbZHYsZ3JvdXBdXS5kcm9wbmEoKTsgcmVzID0gcGFpcndpc2VfdHVrZXloc2QoZW5kb2c9ZFtkdl0uYXN0eXBlKGZsb2F0KSwgZ3JvdXBzPWRbZ3JvdXBdLmFzdHlwZShzdHIpKQogICAgICAgIHJldHVybiBwZC5EYXRhRnJhbWUocmVzLl9yZXN1bHRzX3RhYmxlLmRhdGFbMTpdLCBjb2x1bW5zPXJlcy5fcmVzdWx0c190YWJsZS5kYXRhWzBdKQogICAgZGVmIGFuY292YShzZWxmLCBkdjogc3RyLCBmYWN0b3I6IHN0ciwgY292YXJzOiBMaXN0W3N0cl0pOgogICAgICAgIGRhdGEgPSBzZWxmLmRmLmRyb3BuYShzdWJzZXQ9W2R2LGZhY3Rvcl0rY292YXJzKTsgbSA9IHNtZi5vbHMoZiJ7ZHZ9IH4gQyh7ZmFjdG9yfSkgKyAiICsgIiArICIuam9pbihjb3ZhcnMpLCBkYXRhPWRhdGEpLmZpdCgpOyByZXR1cm4gbQogICAgZGVmIG1hbm53aGl0bmV5KHNlbGYsIGR2OiBzdHIsIGdyb3VwOiBzdHIsIGcxLCBnMik6CiAgICAgICAgeDEgPSBzZWxmLmRmLmxvY1tzZWxmLmRmW2dyb3VwXT09ZzEsIGR2XS5kcm9wbmEoKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgeDIgPSBzZWxmLmRmLmxvY1tzZWxmLmRmW2dyb3VwXT09ZzIsIGR2XS5kcm9wbmEoKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgdSxwID0gc3RhdHMubWFubndoaXRuZXl1KHgxLCB4Mik7IHJldHVybiB7IlUiOiB1LCAicCI6IHB9CiAgICBkZWYgd2lsY294b24oc2VsZiwgY29sMTogc3RyLCBjb2wyOiBzdHIpOgogICAgICAgIHgxID0gc2VsZi5kZltjb2wxXS5hc3R5cGUoZmxvYXQpOyB4MiA9IHNlbGYuZGZbY29sMl0uYXN0eXBlKGZsb2F0KQogICAgICAgIG0gPSB+KHgxLmlzbmEoKSB8IHgyLmlzbmEoKSk7IHgxPXgxW21dOyB4Mj14MlttXTsgVyxwID0gc3RhdHMud2lsY294b24oeDEsIHgyKTsgcmV0dXJuIHsiVyI6IFcsICJwIjogcH0KICAgIGRlZiBrcnVza2FsKHNlbGYsIGR2OiBzdHIsIGdyb3VwOiBzdHIpOgogICAgICAgIGdyb3VwcyA9IFtnLmRyb3BuYSgpLmFzdHlwZShmbG9hdCkudmFsdWVzIGZvciBfLGcgaW4gc2VsZi5kZltbZHYsZ3JvdXBdXS5kcm9wbmEoKS5ncm91cGJ5KGdyb3VwKVtkdl1dCiAgICAgICAgSCxwID0gc3RhdHMua3J1c2thbCgqZ3JvdXBzKTsgcmV0dXJuIHsiSCI6IEgsICJwIjogcH0KICAgIGRlZiBmcmllZG1hbihzZWxmLCBjb2xzOiBMaXN0W3N0cl0pOgogICAgICAgIGFycnMgPSBbc2VsZi5kZltjXS5hc3R5cGUoZmxvYXQpLmRyb3BuYSgpLnZhbHVlcyBmb3IgYyBpbiBjb2xzXTsgbSA9IG1pbihtYXAobGVuLCBhcnJzKSk7IGFycnMgPSBbYVs6bV0gZm9yIGEgaW4gYXJyc10KICAgICAgICBRLHAgPSBzdGF0cy5mcmllZG1hbmNoaXNxdWFyZSgqYXJycyk7IHJldHVybiB7IlEiOiBRLCAicCI6IHB9CgogICAgCiAgICBkZWYgY29ycmVsYXRpb25zKHNlbGYsIGNvbHM6IExpc3Rbc3RyXSwgbWV0aG9kPSdwZWFyc29uJyk6IHJldHVybiBzZWxmLmRmW2NvbHNdLmNvcnIobWV0aG9kPW1ldGhvZCkKCiAgICAKICAgIGRlZiBvbHMoc2VsZiwgZHY6IHN0ciwgcHJlZGljdG9yczogTGlzdFtzdHJdKToKICAgICAgICBkYXRhID0gc2VsZi5kZi5kcm9wbmEoc3Vic2V0PVtkdl0rcHJlZGljdG9ycyk7IHJldHVybiBzbWYub2xzKGYie2R2fSB+ICIgKyAiICsgIi5qb2luKHByZWRpY3RvcnMpLCBkYXRhPWRhdGEpLmZpdCgpCiAgICBkZWYgbG9naXQoc2VsZiwgZHY6IHN0ciwgcHJlZGljdG9yczogTGlzdFtzdHJdKToKICAgICAgICBkYXRhID0gc2VsZi5kZi5kcm9wbmEoc3Vic2V0PVtkdl0rcHJlZGljdG9ycyk7IHJldHVybiBzbWYubG9naXQoZiJ7ZHZ9IH4gIiArICIgKyAiLmpvaW4ocHJlZGljdG9ycyksIGRhdGE9ZGF0YSkuZml0KGRpc3A9RmFsc2UpCiAgICBkZWYgbWxvZ2l0KHNlbGYsIGR2OiBzdHIsIHByZWRpY3RvcnM6IExpc3Rbc3RyXSk6CiAgICAgICAgZGF0YSA9IHNlbGYuZGYuZHJvcG5hKHN1YnNldD1bZHZdK3ByZWRpY3RvcnMpOyByZXR1cm4gc21mLm1ubG9naXQoZiJ7ZHZ9IH4gIiArICIgKyAiLmpvaW4ocHJlZGljdG9ycyksIGRhdGE9ZGF0YSkuZml0KG1ldGhvZD0nbmV3dG9uJywgbWF4aXRlcj0xMDAsIGRpc3A9RmFsc2UpCiAgICBkZWYgb3JkaW5hbChzZWxmLCBkdjogc3RyLCBwcmVkaWN0b3JzOiBMaXN0W3N0cl0pOgogICAgICAgIGZyb20gc3RhdHNtb2RlbHMubWlzY21vZGVscy5vcmRpbmFsX21vZGVsIGltcG9ydCBPcmRlcmVkTW9kZWwKICAgICAgICBkYXRhID0gc2VsZi5kZi5kcm9wbmEoc3Vic2V0PVtkdl0rcHJlZGljdG9ycyk7IHJldHVybiBPcmRlcmVkTW9kZWwoZGF0YVtkdl0sIHNtLmFkZF9jb25zdGFudChkYXRhW3ByZWRpY3RvcnNdKSwgZGlzdHI9J2xvZ2l0JykuZml0KG1ldGhvZD0nYmZncycsIGRpc3A9RmFsc2UpCgogICAgZGVmIGdsbShzZWxmLCBkdjogc3RyLCBwcmVkaWN0b3JzOiBMaXN0W3N0cl0sIGZhbWlseTogc3RyKToKICAgICAgICBmYW0gPSB7J3BvaXNzb24nOiBzbS5mYW1pbGllcy5Qb2lzc29uKCksICduZWdiaW4nOiBzbS5mYW1pbGllcy5OZWdhdGl2ZUJpbm9taWFsKCksICdnYW1tYSc6IHNtLmZhbWlsaWVzLkdhbW1hKGxpbms9c20uZ2VubW9kLmZhbWlsaWVzLmxpbmtzLmxvZygpKX1bZmFtaWx5XQogICAgICAgIGRhdGEgPSBzZWxmLmRmLmRyb3BuYShzdWJzZXQ9W2R2XStwcmVkaWN0b3JzKTsgcmV0dXJuIHNtZi5nbG0oZiJ7ZHZ9IH4gIiArICIgKyAiLmpvaW4ocHJlZGljdG9ycyksIGRhdGE9ZGF0YSwgZmFtaWx5PWZhbSkuZml0KCkKCiAgICAKICAgIGRlZiBsbW0oc2VsZiwgZHY6IHN0ciwgcHJlZGljdG9yczogTGlzdFtzdHJdLCBncm91cDogc3RyKToKICAgICAgICBkYXRhID0gc2VsZi5kZi5kcm9wbmEoc3Vic2V0PVtkdixncm91cF0rcHJlZGljdG9ycyk7IHJldHVybiBzbWYubWl4ZWRsbShmIntkdn0gfiAiICsgIiArICIuam9pbihwcmVkaWN0b3JzKSwgZGF0YT1kYXRhLCBncm91cHM9ZGF0YVtncm91cF0pLmZpdCgpCgogICAgCiAgICBkZWYgYW5vdmFfcm0oc2VsZiwgZHY6IHN0ciwgc3ViamVjdDogc3RyLCB3aXRoaW46IHN0cik6CiAgICAgICAgZGF0YSA9IHNlbGYuZGYuZHJvcG5hKHN1YnNldD1bZHYsIHN1YmplY3QsIHdpdGhpbl0pLmNvcHkoKTsgcmVzID0gQW5vdmFSTShkYXRhLCBkZXB2YXI9ZHYsIHN1YmplY3Q9c3ViamVjdCwgd2l0aGluPVt3aXRoaW5dKS5maXQoKTsgcmV0dXJuIHJlcy5hbm92YV90YWJsZQogICAgZGVmIG1hbm92YShzZWxmLCB5X2NvbHM6IExpc3Rbc3RyXSwgcHJlZGljdG9yczogTGlzdFtzdHJdKToKICAgICAgICBkYXRhID0gc2VsZi5kZi5kcm9wbmEoc3Vic2V0PXlfY29scytwcmVkaWN0b3JzKS5jb3B5KCk7IHkgPSAiICsgIi5qb2luKHlfY29scyk7IHggPSAiICsgIi5qb2luKHByZWRpY3RvcnMpIGlmIHByZWRpY3RvcnMgZWxzZSAiMSIKICAgICAgICByZXR1cm4gTUFOT1ZBLmZyb21fZm9ybXVsYShmInt5fSB+IHt4fSIsIGRhdGE9ZGF0YSkubXZfdGVzdCgpCgogICAgCiAgICBkZWYgY3JvbmJhY2hfYWxwaGEoc2VsZiwgY29sczogTGlzdFtzdHJdKToKICAgICAgICBYID0gc2VsZi5kZltjb2xzXS5kcm9wbmEoKS5hc3R5cGUoZmxvYXQpLnZhbHVlcwogICAgICAgIGsgPSBYLnNoYXBlWzFdOyB2YXJfc3VtID0gWC52YXIoYXhpcz0wLCBkZG9mPTEpLnN1bSgpOyB0b3RhbF92YXIgPSBYLnN1bShheGlzPTEpLnZhcihkZG9mPTEpCiAgICAgICAgcmV0dXJuIChrLyhrLTEuMCkpKigxIC0gdmFyX3N1bS90b3RhbF92YXIpCgogICAgZGVmIGthcHBhKHNlbGYsIGNvbDE6IHN0ciwgY29sMjogc3RyKToKICAgICAgICBpZiBub3QgY29sMSBvciBub3QgY29sMjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiU2VsZWN0IGJvdGggWSBhbmQgWCBjb2x1bW5zIGZvciBDb2hlbidzIGthcHBhLiIpCiAgICAgICAgZm9yIGMgaW4gKGNvbDEsIGNvbDIpOgogICAgICAgICAgICBpZiBjIG5vdCBpbiBzZWxmLmRmLmNvbHVtbnM6CiAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiQ29sdW1uICd7Y30nIG5vdCBpbiBkYXRhc2V0LiBBdmFpbGFibGU6IHtsaXN0KHNlbGYuZGYuY29sdW1ucyl9IikKICAgICAgICBhID0gc2VsZi5kZltjb2wxXS5hc3R5cGUoc3RyKTsgYiA9IHNlbGYuZGZbY29sMl0uYXN0eXBlKHN0cikKICAgICAgICBuID0gbWluKGxlbihhKSwgbGVuKGIpKTsgYT1hLmlsb2NbOm5dOyBiPWIuaWxvY1s6bl0KICAgICAgICByZXR1cm4gY29oZW5fa2FwcGFfc2NvcmUoYSwgYikKCiAgICBkZWYgd2VpZ2h0ZWRfa2FwcGEoc2VsZiwgY29sMTogc3RyLCBjb2wyOiBzdHIsIHdlaWdodHM6IHN0ciA9ICJsaW5lYXIiKToKICAgICAgICBpZiBub3QgY29sMSBvciBub3QgY29sMjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiU2VsZWN0IGJvdGggWSBhbmQgWCBjb2x1bW5zIGZvciB3ZWlnaHRlZCBrYXBwYS4iKQogICAgICAgIGZvciBjIGluIChjb2wxLCBjb2wyKToKICAgICAgICAgICAgaWYgYyBub3QgaW4gc2VsZi5kZi5jb2x1bW5zOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIkNvbHVtbiAne2N9JyBub3QgaW4gZGF0YXNldC4iKQogICAgICAgIGEgPSBzZWxmLmRmW2NvbDFdOyBiID0gc2VsZi5kZltjb2wyXQogICAgICAgIG1hc2sgPSB+KGEuaXNuYSgpIHwgYi5pc25hKCkpOyBhPWFbbWFza107IGI9YlttYXNrXQogICAgICAgIGFfbnVtID0gcGQudG9fbnVtZXJpYyhhLCBlcnJvcnM9J2NvZXJjZScpCiAgICAgICAgYl9udW0gPSBwZC50b19udW1lcmljKGIsIGVycm9ycz0nY29lcmNlJykKICAgICAgICBpZiBhX251bS5ub3RuYSgpLmFsbCgpIGFuZCBiX251bS5ub3RuYSgpLmFsbCgpOgogICAgICAgICAgICByZXR1cm4gY29oZW5fa2FwcGFfc2NvcmUoYV9udW0sIGJfbnVtLCB3ZWlnaHRzPXdlaWdodHMpCiAgICAgICAgbGUgPSBMYWJlbEVuY29kZXIoKQogICAgICAgIGxlLmZpdChwZC5jb25jYXQoW2EuYXN0eXBlKHN0ciksIGIuYXN0eXBlKHN0cildLCBpZ25vcmVfaW5kZXg9VHJ1ZSkpCiAgICAgICAgYV9lbmMgPSBsZS50cmFuc2Zvcm0oYS5hc3R5cGUoc3RyKSk7IGJfZW5jID0gbGUudHJhbnNmb3JtKGIuYXN0eXBlKHN0cikpCiAgICAgICAgcmV0dXJuIGNvaGVuX2thcHBhX3Njb3JlKGFfZW5jLCBiX2VuYywgd2VpZ2h0cz13ZWlnaHRzKQoKICAgIGRlZiBzY290dF9waShzZWxmLCBjb2wxOiBzdHIsIGNvbDI6IHN0cik6CiAgICAgICAgaWYgbm90IGNvbDEgb3Igbm90IGNvbDI6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlNlbGVjdCBib3RoIGNvbHVtbnMgZm9yIFNjb3R0J3MgcGkuIikKICAgICAgICBmb3IgYyBpbiAoY29sMSwgY29sMik6CiAgICAgICAgICAgIGlmIGMgbm90IGluIHNlbGYuZGYuY29sdW1uczoKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJDb2x1bW4gJ3tjfScgbm90IGluIGRhdGFzZXQuIEF2YWlsYWJsZToge2xpc3Qoc2VsZi5kZi5jb2x1bW5zKX0iKQogICAgICAgIGEgPSBzZWxmLmRmW2NvbDFdLmFzdHlwZShzdHIpOyBiID0gc2VsZi5kZltjb2wyXS5hc3R5cGUoc3RyKQogICAgICAgIG4gPSBtaW4obGVuKGEpLCBsZW4oYikpOyBhPWEuaWxvY1s6bl07IGI9Yi5pbG9jWzpuXQogICAgICAgIHBvID0gZmxvYXQoKGEudmFsdWVzID09IGIudmFsdWVzKS5tZWFuKCkpCiAgICAgICAgcG9vbGVkID0gcGQuY29uY2F0KFthLCBiXSwgaWdub3JlX2luZGV4PVRydWUpCiAgICAgICAgcCA9IHBvb2xlZC52YWx1ZV9jb3VudHMobm9ybWFsaXplPVRydWUpCiAgICAgICAgcGUgPSBmbG9hdCgocCoqMikuc3VtKCkpCiAgICAgICAgaWYgcGUgPT0gMS4wOiByZXR1cm4gMS4wCiAgICAgICAgcmV0dXJuIChwbyAtIHBlKSAvICgxLjAgLSBwZSkKCiAgICBkZWYgZmxlaXNzX2thcHBhKHNlbGYsIGNvbHM6IExpc3Rbc3RyXSk6CiAgICAgICAgcmF0ZXJzID0gc2VsZi5kZltjb2xzXS5kcm9wbmEoKS5hc3R5cGUoc3RyKQogICAgICAgIGNhdHMgPSBzb3J0ZWQocGQudW5pcXVlKHJhdGVycy52YWx1ZXMucmF2ZWwoKSkpCiAgICAgICAgY2F0X3RvX2lkeCA9IHtjOmkgZm9yIGksYyBpbiBlbnVtZXJhdGUoY2F0cyl9CiAgICAgICAgdGFibGUgPSBucC56ZXJvcygobGVuKHJhdGVycyksIGxlbihjYXRzKSksIGR0eXBlPWludCkKICAgICAgICBmb3IgaSwgKF8sIHJvdykgaW4gZW51bWVyYXRlKHJhdGVycy5pdGVycm93cygpKToKICAgICAgICAgICAgY29kZXMgPSByb3cubWFwKGNhdF90b19pZHgpLmRyb3BuYSgpLmFzdHlwZShpbnQpLnZhbHVlcwogICAgICAgICAgICBmb3IgY29kZSBpbiBjb2RlczoKICAgICAgICAgICAgICAgIHRhYmxlW2ksIGNvZGVdICs9IDEKICAgICAgICBrYXBwYSA9IGZsb2F0KHNtX2ZsZWlzc19rYXBwYSh0YWJsZSwgbWV0aG9kPSdmbGVpc3MnKSkKICAgICAgICByZXR1cm4ga2FwcGEsIGxlbihyYXRlcnMpLCBsZW4oY29scyksIGxlbihjYXRzKQoKICAgIGRlZiBpY2Moc2VsZiwgY29sczogTGlzdFtzdHJdKToKICAgICAgICBpZiBub3QgSEFTX1BJTkdPVUlOOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoInBpbmdvdWluIGlzIG5vdCBpbnN0YWxsZWQuIFJ1bjogcGlwIGluc3RhbGwgcGluZ291aW4iKQogICAgICAgIGlmIGNvbHMgaXMgTm9uZSBvciBsZW4oY29scykgPCAyOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIGF0IGxlYXN0IDIgcmF0ZXIgY29sdW1ucyBmb3IgSUNDLiIpCiAgICAgICAgZm9yIGMgaW4gY29sczoKICAgICAgICAgICAgaWYgYyBub3QgaW4gc2VsZi5kZi5jb2x1bW5zOgogICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIkNvbHVtbiAne2N9JyBub3QgZm91bmQgaW4gZGF0YS4iKQogICAgICAgIGRmdyA9IHNlbGYuZGZbY29sc10uY29weSgpLmRyb3BuYShob3c9ImFueSIpCiAgICAgICAgaWYgZGZ3LmVtcHR5OiByYWlzZSBWYWx1ZUVycm9yKCJObyBjb21wbGV0ZSByb3dzIGFjcm9zcyB0aGUgc2VsZWN0ZWQgcmF0ZXIgY29sdW1ucy4iKQogICAgICAgIGRmdyA9IGRmdy5yZXNldF9pbmRleCgpLnJlbmFtZShjb2x1bW5zPXsiaW5kZXgiOiAic3ViamVjdCJ9KQogICAgICAgIGxvbmcgPSBkZncubWVsdChpZF92YXJzPSJzdWJqZWN0IiwgdmFsdWVfdmFycz1jb2xzLCB2YXJfbmFtZT0icmF0ZXIiLCB2YWx1ZV9uYW1lPSJyYXRpbmciKQogICAgICAgIG91dCA9IHBnLmljYyhkYXRhPWxvbmcsIHRhcmdldHM9InN1YmplY3QiLCByYXRlcnM9InJhdGVyIiwgcmF0aW5ncz0icmF0aW5nIikKICAgICAgICBwcmVmZXIgPSBbIklDQzEiLCJJQ0MxayIsIklDQzIiLCJJQ0MyayIsIklDQzMiLCJJQ0MzayJdOyBvdXRbIm9yZGVyIl0gPSBvdXRbIlR5cGUiXS5hcHBseShsYW1iZGEgdDogcHJlZmVyLmluZGV4KHQpIGlmIHQgaW4gcHJlZmVyIGVsc2UgOTk5KQogICAgICAgIHJldHVybiBvdXQuc29ydF92YWx1ZXMoIm9yZGVyIikuZHJvcChjb2x1bW5zPVsib3JkZXIiXSkucmVzZXRfaW5kZXgoZHJvcD1UcnVlKQoKICAgIGRlZiBrcmlwcF9hbHBoYShzZWxmLCBjb2xzOiBMaXN0W3N0cl0sIGxldmVsOiBzdHIgPSAibm9taW5hbCIpOgogICAgICAgIGlmIG5vdCBIQVNfS1JJUFA6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigia3JpcHBlbmRvcmZmIHBhY2thZ2UgaXMgbm90IGluc3RhbGxlZC4gUnVuOiBwaXAgaW5zdGFsbCBrcmlwcGVuZG9yZmYiKQogICAgICAgIGlmIGNvbHMgaXMgTm9uZSBvciBsZW4oY29scykgPCAyOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIGF0IGxlYXN0IDIgcmF0ZXIgY29sdW1ucyBmb3IgS3JpcHBlbmRvcmZm4oCZcyBhbHBoYS4iKQogICAgICAgIGZvciBjIGluIGNvbHM6CiAgICAgICAgICAgIGlmIGMgbm90IGluIHNlbGYuZGYuY29sdW1uczoKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJDb2x1bW4gJ3tjfScgbm90IGZvdW5kIGluIGRhdGEuIikKICAgICAgICBtYXQgPSBzZWxmLmRmW2NvbHNdLlQudmFsdWVzLnRvbGlzdCgpCiAgICAgICAgYWxwaGEgPSBrZC5hbHBoYShyZWxpYWJpbGl0eV9kYXRhPW1hdCwgbGV2ZWxfb2ZfbWVhc3VyZW1lbnQ9bGV2ZWwpCiAgICAgICAgcmV0dXJuIGZsb2F0KGFscGhhKQoKICAgIAogICAgZGVmIHN1cnZpdmFsKHNlbGYsIHRpbWU6IHN0ciwgZXZlbnQ6IHN0ciwgZ3JvdXA6IE9wdGlvbmFsW3N0cl0sIGNvdmFyczogTGlzdFtzdHJdLCByZXBvcnQpOgogICAgICAgIGlmIG5vdCBIQVNfTElGRUxJTkVTOiByZXBvcnQuYWRkX2luZm8oIlN1cnZpdmFsIHVuYXZhaWxhYmxlIiwiSW5zdGFsbCBsaWZlbGluZXMuIik7IHJldHVybgogICAgICAgIGZyb20gbGlmZWxpbmVzIGltcG9ydCBLYXBsYW5NZWllckZpdHRlciwgQ294UEhGaXR0ZXIsIFdlaWJ1bGxBRlRGaXR0ZXIKICAgICAgICBkZiA9IHNlbGYuZGZbW3RpbWUsIGV2ZW50XSArIChbZ3JvdXBdIGlmIGdyb3VwIGVsc2UgW10pICsgY292YXJzXS5kcm9wbmEoKQogICAgICAgIGttZiA9IEthcGxhbk1laWVyRml0dGVyKCkKICAgICAgICBpZiBncm91cCBhbmQgZGZbZ3JvdXBdLm51bmlxdWUoKT4xOgogICAgICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCkKICAgICAgICAgICAgZm9yIGcsIHN1YiBpbiBkZi5ncm91cGJ5KGdyb3VwKToKICAgICAgICAgICAgICAgIGttZi5maXQoc3ViW3RpbWVdLCBldmVudF9vYnNlcnZlZD1zdWJbZXZlbnRdLCBsYWJlbD1zdHIoZykpOyBrbWYucGxvdF9zdXJ2aXZhbF9mdW5jdGlvbigpCiAgICAgICAgICAgIHBsdC50aXRsZSgiS2FwbGFu4oCTTWVpZXIiKTsgcmVwb3J0LmFkZF9maWd1cmUoZmlnLCAiS00gYnkgZ3JvdXAiKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGttZi5maXQoZGZbdGltZV0sIGV2ZW50X29ic2VydmVkPWRmW2V2ZW50XSwgbGFiZWw9IkFsbCIpOyBheCA9IGttZi5wbG90X3N1cnZpdmFsX2Z1bmN0aW9uKCk7IGF4LnNldF90aXRsZSgiS2FwbGFu4oCTTWVpZXIiKQogICAgICAgICAgICByZXBvcnQuYWRkX2ZpZ3VyZShheC5nZXRfZmlndXJlKCksICJLYXBsYW7igJNNZWllciIpCiAgICAgICAgaWYgY292YXJzOgogICAgICAgICAgICBjcGggPSBDb3hQSEZpdHRlcigpOyBjcGguZml0KGRmW1t0aW1lLGV2ZW50XStjb3ZhcnNdLCBkdXJhdGlvbl9jb2w9dGltZSwgZXZlbnRfY29sPWV2ZW50KQogICAgICAgICAgICByZXBvcnQuYWRkX3RhYmxlKGNwaC5zdW1tYXJ5LnJlc2V0X2luZGV4KCksICJDb3ggUEgiKQogICAgICAgICAgICBhZnQgPSBXZWlidWxsQUZURml0dGVyKCk7IGFmdC5maXQoZGZbW3RpbWUsZXZlbnRdK2NvdmFyc10sIGR1cmF0aW9uX2NvbD10aW1lLCBldmVudF9jb2w9ZXZlbnQpCiAgICAgICAgICAgIHJlcG9ydC5hZGRfdGFibGUoYWZ0LnN1bW1hcnkucmVzZXRfaW5kZXgoKSwgIldlaWJ1bGwgQUZUIikKCiAgICBkZWYgZGlhZ25vc3RpY19jdXJ2ZXMoc2VsZiwgeTogc3RyLCBzY29yZTogc3RyLCByZXBvcnQpOgogICAgICAgIGQgPSBzZWxmLmRmW1t5LCBzY29yZV1dLmRyb3BuYSgpCiAgICAgICAgaWYgZFt5XS5udW5pcXVlKCkhPTI6IHJlcG9ydC5hZGRfaW5mbygiRGlhZ25vc3RpYyBlcnJvciIsIk91dGNvbWUgbXVzdCBiZSBiaW5hcnkuIik7IHJldHVybgogICAgICAgIGZwciwgdHByLCBfID0gcm9jX2N1cnZlKGRbeV0sIGRbc2NvcmVdKTsgcHJlYywgcmVjLCBfID0gcHJlY2lzaW9uX3JlY2FsbF9jdXJ2ZShkW3ldLCBkW3Njb3JlXSkKICAgICAgICBmaWcxPXBsdC5maWd1cmUoKTsgcGx0LnBsb3QoZnByLHRwcik7IHBsdC5wbG90KFswLDFdLFswLDFdLCctLScpOyBwbHQueGxhYmVsKCJGUFIiKTsgcGx0LnlsYWJlbCgiVFBSIik7IHBsdC50aXRsZSgiUk9DIikKICAgICAgICBmaWcyPXBsdC5maWd1cmUoKTsgcGx0LnBsb3QocmVjLHByZWMpOyBwbHQueGxhYmVsKCJSZWNhbGwiKTsgcGx0LnlsYWJlbCgiUHJlY2lzaW9uIik7IHBsdC50aXRsZSgiUFIgY3VydmUiKQogICAgICAgIHJlcG9ydC5hZGRfZmlndXJlKGZpZzEsICJST0MiKTsgcmVwb3J0LmFkZF9maWd1cmUoZmlnMiwgIlBSIik7IGJyaWVyID0gZmxvYXQobnAubWVhbigoZFtzY29yZV0tZFt5XSkqKjIpKTsgcmVwb3J0LmFkZF9rdigiQnJpZXIgc2NvcmUiLCB7IkJyaWVyIjogcm91bmQoYnJpZXIsNCl9KQoKICAgIGRlZiBhcmltYV9mb3JlY2FzdChzZWxmLCB0aW1lX2NvbDogc3RyLCB2YWx1ZV9jb2w6IHN0ciwgc3RlcHM6IGludCwgcmVwb3J0KToKICAgICAgICBmcm9tIHN0YXRzbW9kZWxzLnRzYS5hcmltYS5tb2RlbCBpbXBvcnQgQVJJTUEKICAgICAgICBkZiA9IHNlbGYuZGZbW3RpbWVfY29sLCB2YWx1ZV9jb2xdXS5kcm9wbmEoKS5jb3B5KCkuc29ydF92YWx1ZXModGltZV9jb2wpCiAgICAgICAgeSA9IHBkLlNlcmllcyhkZlt2YWx1ZV9jb2xdLnZhbHVlcywgaW5kZXg9cGQudG9fZGF0ZXRpbWUoZGZbdGltZV9jb2xdKSkKICAgICAgICBtb2RlbCA9IEFSSU1BKHksIG9yZGVyPSgxLDAsMSkpLmZpdCgpCiAgICAgICAgZmMgPSBtb2RlbC5nZXRfZm9yZWNhc3Qoc3RlcHM9c3RlcHMpOyBwcmVkPWZjLnByZWRpY3RlZF9tZWFuOyBjaT1mYy5jb25mX2ludCgpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBmcmVxID0gcGQuaW5mZXJfZnJlcSh5LmluZGV4KTsgZnV0dXJlX2luZGV4ID0gcGQuZGF0ZV9yYW5nZSh5LmluZGV4Wy0xXSwgcGVyaW9kcz1zdGVwcysxLCBmcmVxPWZyZXEgb3IgJ0QnKVsxOl0KICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBmdXR1cmVfaW5kZXggPSBwZC5kYXRlX3JhbmdlKHkuaW5kZXhbLTFdLCBwZXJpb2RzPXN0ZXBzKzEsIGZyZXE9J0QnKVsxOl0KICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCk7IHBsdC5wbG90KHkuaW5kZXgsIHkudmFsdWVzLCBsYWJlbD0iT2JzZXJ2ZWQiKTsgcGx0LnBsb3QoZnV0dXJlX2luZGV4LCBwcmVkLnZhbHVlcywgbGFiZWw9IkZvcmVjYXN0IikKICAgICAgICBwbHQuZmlsbF9iZXR3ZWVuKGZ1dHVyZV9pbmRleCwgY2kuaWxvY1s6LDBdLnZhbHVlcywgY2kuaWxvY1s6LDFdLnZhbHVlcywgYWxwaGE9MC4yKTsgcGx0LmxlZ2VuZCgpOyBwbHQudGl0bGUoIkFSSU1BIEZvcmVjYXN0IikKICAgICAgICByZXBvcnQuYWRkX2ZpZ3VyZShmaWcsICJBUklNQSIpCgogICAgZGVmIGV0c19mb3JlY2FzdChzZWxmLCB0aW1lX2NvbDogc3RyLCB2YWx1ZV9jb2w6IHN0ciwgc3RlcHM6IGludCwgcmVwb3J0LCBzZWFzb25hbD1Ob25lKToKICAgICAgICBmcm9tIHN0YXRzbW9kZWxzLnRzYS5ob2x0d2ludGVycyBpbXBvcnQgRXhwb25lbnRpYWxTbW9vdGhpbmcKICAgICAgICBkZiA9IHNlbGYuZGZbW3RpbWVfY29sLCB2YWx1ZV9jb2xdXS5kcm9wbmEoKS5jb3B5KCkuc29ydF92YWx1ZXModGltZV9jb2wpCiAgICAgICAgeSA9IHBkLlNlcmllcyhkZlt2YWx1ZV9jb2xdLnZhbHVlcywgaW5kZXg9cGQudG9fZGF0ZXRpbWUoZGZbdGltZV9jb2xdKSkKICAgICAgICBtb2RlbCA9IEV4cG9uZW50aWFsU21vb3RoaW5nKHksIHRyZW5kPSdhZGQnLCBzZWFzb25hbD1zZWFzb25hbCwgc2Vhc29uYWxfcGVyaW9kcz0oMTIgaWYgc2Vhc29uYWwgZWxzZSBOb25lKSkuZml0KCkKICAgICAgICBwcmVkID0gbW9kZWwuZm9yZWNhc3Qoc3RlcHMpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBmdXR1cmVfaW5kZXggPSBwZC5kYXRlX3JhbmdlKHkuaW5kZXhbLTFdLCBwZXJpb2RzPXN0ZXBzKzEsIGZyZXE9cGQuaW5mZXJfZnJlcSh5LmluZGV4KSBvciAnRCcpWzE6XQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGZ1dHVyZV9pbmRleCA9IHBkLmRhdGVfcmFuZ2UoeS5pbmRleFstMV0sIHBlcmlvZHM9c3RlcHMrMSwgZnJlcT0nRCcpWzE6XQogICAgICAgIGZpZyA9IHBsdC5maWd1cmUoKTsgcGx0LnBsb3QoeS5pbmRleCwgeS52YWx1ZXMsIGxhYmVsPSJPYnNlcnZlZCIpOyBwbHQucGxvdChmdXR1cmVfaW5kZXgsIHByZWQudmFsdWVzLCBsYWJlbD0iRVRTIEZvcmVjYXN0Iik7IHBsdC5sZWdlbmQoKQogICAgICAgIHJlcG9ydC5hZGRfZmlndXJlKGZpZywgIkVUUyBGb3JlY2FzdCIpCgogICAgZGVmIG1ldGFfZGwoc2VsZiwgeWk6IHN0ciwgdmk6IHN0ciwgYnk6IE9wdGlvbmFsW3N0cl0sIHJlcG9ydCk6CiAgICAgICAgZCA9IHNlbGYuZGZbW3lpLHZpXSArIChbYnldIGlmIGJ5IGVsc2UgW10pXS5kcm9wbmEoKS5jb3B5KCkKICAgICAgICBkZWYgYmxvY2soc3ViLCBsYWJlbCk6CiAgICAgICAgICAgIHkgPSBzdWJbeWldLnZhbHVlczsgdiA9IHN1Ylt2aV0udmFsdWVzOyB3ID0gMS92CiAgICAgICAgICAgIHlmaXggPSBucC5zdW0odyp5KS9ucC5zdW0odyk7IFEgPSBucC5zdW0odyooeS15Zml4KSoqMik7IGRmcT1sZW4oeSktMQogICAgICAgICAgICBDID0gbnAuc3VtKHcpIC0gKG5wLnN1bSh3KioyKS9ucC5zdW0odykpOyB0YXUyID0gbWF4KDAuMCwgKFEtZGZxKS9DKSBpZiBkZnE+MCBlbHNlIDAuMAogICAgICAgICAgICB3ciA9IDEvKHYrdGF1Mik7IHlyID0gbnAuc3VtKHdyKnkpL25wLnN1bSh3cik7IHNlID0gbnAuc3FydCgxL25wLnN1bSh3cikpOyBsbz15ci0xLjk2KnNlOyBoaT15cisxLjk2KnNlCiAgICAgICAgICAgIEkyID0gbWF4KDAuMCwoUS1kZnEpL1EpKjEwMCBpZiBRPjAgZWxzZSAwLjAKICAgICAgICAgICAgcmV0dXJuIHsiR3JvdXAiOiBsYWJlbCwgInlfcmFuZG9tIjogeXIsICJDSSBsb3ciOiBsbywgIkNJIGhpZ2giOiBoaSwgInRhdV4yIjogdGF1MiwgIlEiOiBRLCAiZGYiOiBkZnEsICJJXjIlIjogSTJ9CiAgICAgICAgcm93cz1bXQogICAgICAgIGlmIGJ5OgogICAgICAgICAgICBmb3IgZyxzdWIgaW4gZC5ncm91cGJ5KGJ5KTogcm93cy5hcHBlbmQoYmxvY2soc3ViLCBzdHIoZykpKQogICAgICAgIHJvd3MuYXBwZW5kKGJsb2NrKGQsICJPdmVyYWxsIikpCiAgICAgICAgcmVwb3J0LmFkZF90YWJsZShwZC5EYXRhRnJhbWUocm93cykucm91bmQoNiksICJSYW5kb20tZWZmZWN0cyBtZXRhLWFuYWx5c2lzIChETCkiKQoKICAgIGRlZiBwY2Eoc2VsZiwgY29sczogTGlzdFtzdHJdLCBuX2NvbXBvbmVudHM6IGludCwgcmVwb3J0KToKICAgICAgICBYID0gc2VsZi5kZltjb2xzXS5kcm9wbmEoKS5hc3R5cGUoZmxvYXQpLnZhbHVlczsgWCA9IFN0YW5kYXJkU2NhbGVyKCkuZml0X3RyYW5zZm9ybShYKQogICAgICAgIHAgPSBQQ0Eobl9jb21wb25lbnRzPW5fY29tcG9uZW50cykuZml0KFgpCiAgICAgICAgY29tcCA9IHBkLkRhdGFGcmFtZShwLmNvbXBvbmVudHNfLCBjb2x1bW5zPWNvbHMpOyBleHBsID0gcGQuRGF0YUZyYW1lKHsiUEMiOiBucC5hcmFuZ2UoMSwgbl9jb21wb25lbnRzKzEpLCAiRXhwbGFpbmVkVmFyIjogcC5leHBsYWluZWRfdmFyaWFuY2VfcmF0aW9ffSkKICAgICAgICByZXBvcnQuYWRkX3RhYmxlKGV4cGwucm91bmQoNCksICJQQ0EgZXhwbGFpbmVkIHZhcmlhbmNlIik7IHJlcG9ydC5hZGRfdGFibGUoY29tcC5yb3VuZCg0KSwgIkNvbXBvbmVudCBsb2FkaW5ncyIpCiAgICBkZWYgZWZhKHNlbGYsIGNvbHM6IExpc3Rbc3RyXSwgbl9mYWN0b3JzOiBpbnQsIHJlcG9ydCk6CiAgICAgICAgaWYgbm90IEhBU19GQTogcmVwb3J0LmFkZF9pbmZvKCJFRkEgdW5hdmFpbGFibGUiLCJJbnN0YWxsIGZhY3Rvci1hbmFseXplci4iKTsgcmV0dXJuCiAgICAgICAgWCA9IHNlbGYuZGZbY29sc10uZHJvcG5hKCkuYXN0eXBlKGZsb2F0KS52YWx1ZXM7IGZyb20gZmFjdG9yX2FuYWx5emVyIGltcG9ydCBGYWN0b3JBbmFseXplcgogICAgICAgIGZhID0gRmFjdG9yQW5hbHl6ZXIobl9mYWN0b3JzPW5fZmFjdG9ycywgcm90YXRpb249J3ZhcmltYXgnKTsgZmEuZml0KFgpCiAgICAgICAgbG9hZCA9IHBkLkRhdGFGcmFtZShmYS5sb2FkaW5nc18sIGNvbHVtbnM9W2YiRntpKzF9IiBmb3IgaSBpbiByYW5nZShuX2ZhY3RvcnMpXSwgaW5kZXg9Y29scyk7IHJlcG9ydC5hZGRfdGFibGUobG9hZC5yb3VuZCg0KSwgIkVGQSB2YXJpbWF4IGxvYWRpbmdzIikKICAgIGRlZiBjb3JyZXNwb25kZW5jZShzZWxmLCBhOiBzdHIsIGI6IHN0ciwgcmVwb3J0KToKICAgICAgICBpZiBub3QgSEFTX1BSSU5DRTogcmVwb3J0LmFkZF9pbmZvKCJDb3JyZXNwb25kZW5jZSB1bmF2YWlsYWJsZSIsIkluc3RhbGwgJ3ByaW5jZScuIik7IHJldHVybgogICAgICAgIHRhYiA9IHBkLmNyb3NzdGFiKHNlbGYuZGZbYV0sIHNlbGYuZGZbYl0pOyBpbXBvcnQgcHJpbmNlCiAgICAgICAgY2EgPSBwcmluY2UuQ0Eobl9jb21wb25lbnRzPTIsIG5faXRlcj0xMCwgY29weT1UcnVlLCBjaGVja19pbnB1dD1UcnVlKS5maXQodGFiKQogICAgICAgIHIgPSBjYS5yb3dfY29vcmRpbmF0ZXModGFiKTsgYyA9IGNhLmNvbHVtbl9jb29yZGluYXRlcyh0YWIpCiAgICAgICAgcmVwb3J0LmFkZF90YWJsZShyLnJvdW5kKDQpLCAiUm93IGNvb3JkaW5hdGVzIChDQSkiKTsgcmVwb3J0LmFkZF90YWJsZShjLnJvdW5kKDQpLCAiQ29sdW1uIGNvb3JkaW5hdGVzIChDQSkiKQoKICAgIGRlZiBsZGEoc2VsZiwgZHY6IHN0ciwgcHJlZGljdG9yczogTGlzdFtzdHJdLCByZXBvcnQpOgogICAgICAgIGQgPSBzZWxmLmRmW1tkdl0rcHJlZGljdG9yc10uZHJvcG5hKCk7IHk9ZFtkdl0uYXN0eXBlKHN0cik7IFg9ZFtwcmVkaWN0b3JzXS5hc3R5cGUoZmxvYXQpCiAgICAgICAgbW9kZWwgPSBMREEoKS5maXQoWCx5KTsgYWNjID0gZmxvYXQobW9kZWwuc2NvcmUoWCx5KSk7IHJlcG9ydC5hZGRfa3YoIkxEQSB0cmFpbmluZyBhY2N1cmFjeSIsIHsiQWNjdXJhY3kiOiByb3VuZChhY2MsNCl9KQogICAgZGVmIHRyZWVfY2xzKHNlbGYsIGR2OiBzdHIsIHByZWRpY3RvcnM6IExpc3Rbc3RyXSwgcmVwb3J0LCBjcml0ZXJpb249ImdpbmkiKToKICAgICAgICBmcm9tIHNrbGVhcm4udHJlZSBpbXBvcnQgRGVjaXNpb25UcmVlQ2xhc3NpZmllcgogICAgICAgIGQgPSBzZWxmLmRmW1tkdl0rcHJlZGljdG9yc10uZHJvcG5hKCk7IHk9ZFtkdl0uYXN0eXBlKHN0cik7IFg9ZFtwcmVkaWN0b3JzXS5hc3R5cGUoZmxvYXQpCiAgICAgICAgbW9kZWwgPSBEZWNpc2lvblRyZWVDbGFzc2lmaWVyKHJhbmRvbV9zdGF0ZT0wLCBjcml0ZXJpb249Y3JpdGVyaW9uLCBtaW5fc2FtcGxlc19sZWFmPTUpLmZpdChYLHkpOyBhY2M9ZmxvYXQobW9kZWwuc2NvcmUoWCx5KSkKICAgICAgICByZXBvcnQuYWRkX2t2KCgiQ0hBSUQtbGlrZSB0cmVlIiBpZiBjcml0ZXJpb249PSdlbnRyb3B5JyBlbHNlICJEZWNpc2lvbiB0cmVlIikrIiAodHJhaW5pbmcgYWNjdXJhY3kpIiwgeyJBY2N1cmFjeSI6IHJvdW5kKGFjYyw0KX0pCiAgICBkZWYgcmZfY2xzKHNlbGYsIGR2OiBzdHIsIHByZWRpY3RvcnM6IExpc3Rbc3RyXSwgcmVwb3J0KToKICAgICAgICBmcm9tIHNrbGVhcm4uZW5zZW1ibGUgaW1wb3J0IFJhbmRvbUZvcmVzdENsYXNzaWZpZXIKICAgICAgICBkID0gc2VsZi5kZltbZHZdK3ByZWRpY3RvcnNdLmRyb3BuYSgpOyB5PWRbZHZdLmFzdHlwZShzdHIpOyBYPWRbcHJlZGljdG9yc10uYXN0eXBlKGZsb2F0KQogICAgICAgIG1vZGVsID0gUmFuZG9tRm9yZXN0Q2xhc3NpZmllcihuX2VzdGltYXRvcnM9MjAwLCByYW5kb21fc3RhdGU9MCkuZml0KFgseSk7IGFjYz1mbG9hdChtb2RlbC5zY29yZShYLHkpKQogICAgICAgIHJlcG9ydC5hZGRfa3YoIlJhbmRvbSBmb3Jlc3QgKHRyYWluaW5nIGFjY3VyYWN5KSIsIHsiQWNjdXJhY3kiOiByb3VuZChhY2MsNCl9KQoKICAgIGRlZiBrbWVhbnMoc2VsZiwgY29sczogTGlzdFtzdHJdLCBrOiBpbnQsIHJlcG9ydCk6CiAgICAgICAgWCA9IHNlbGYuZGZbY29sc10uZHJvcG5hKCkuYXN0eXBlKGZsb2F0KS52YWx1ZXM7IFggPSBTdGFuZGFyZFNjYWxlcigpLmZpdF90cmFuc2Zvcm0oWCkKICAgICAgICBrbSA9IEtNZWFucyhuX2NsdXN0ZXJzPWssIG5faW5pdD0xMCwgcmFuZG9tX3N0YXRlPTApLmZpdChYKTsgY2VudGVycyA9IHBkLkRhdGFGcmFtZShrbS5jbHVzdGVyX2NlbnRlcnNfLCBjb2x1bW5zPWNvbHMpCiAgICAgICAgcmVwb3J0LmFkZF90YWJsZShjZW50ZXJzLnJvdW5kKDQpLCAiSy1NZWFucyBjbHVzdGVyIGNlbnRlcnMiKQogICAgZGVmIGFnZ2xvbWVyYXRpdmUoc2VsZiwgY29sczogTGlzdFtzdHJdLCBrOiBpbnQsIHJlcG9ydCk6CiAgICAgICAgWCA9IHNlbGYuZGZbY29sc10uZHJvcG5hKCkuYXN0eXBlKGZsb2F0KS52YWx1ZXM7IFggPSBTdGFuZGFyZFNjYWxlcigpLmZpdF90cmFuc2Zvcm0oWCkKICAgICAgICBhZ2cgPSBBZ2dsb21lcmF0aXZlQ2x1c3RlcmluZyhuX2NsdXN0ZXJzPWspLmZpdChYKTsgbGFiZWxzID0gcGQuU2VyaWVzKGFnZy5sYWJlbHNfLCBuYW1lPSJjbHVzdGVyIikKICAgICAgICByZXBvcnQuYWRkX3RhYmxlKHBkLkRhdGFGcmFtZShsYWJlbHMudmFsdWVfY291bnRzKCkpLCAiQWdnbG9tZXJhdGl2ZSBjbHVzdGVyIHNpemVzIikKICAgIGRlZiBhdXRvX2ttZWFucyhzZWxmLCBjb2xzOiBMaXN0W3N0cl0sIHJlcG9ydCk6CiAgICAgICAgWCA9IHNlbGYuZGZbY29sc10uZHJvcG5hKCkuYXN0eXBlKGZsb2F0KS52YWx1ZXM7IFhzID0gU3RhbmRhcmRTY2FsZXIoKS5maXRfdHJhbnNmb3JtKFgpCiAgICAgICAgYmVzdF9rLCBiZXN0X3MgPSBOb25lLCAtMQogICAgICAgIGZvciBrIGluIHJhbmdlKDIsIDExKToKICAgICAgICAgICAga20gPSBLTWVhbnMobl9jbHVzdGVycz1rLCBuX2luaXQ9MTAsIHJhbmRvbV9zdGF0ZT0wKS5maXQoWHMpCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHMgPSBzaWxob3VldHRlX3Njb3JlKFhzLCBrbS5sYWJlbHNfKQogICAgICAgICAgICAgICAgaWYgcyA+IGJlc3RfczogYmVzdF9zLCBiZXN0X2sgPSBzLCBrCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgcmVwb3J0LmFkZF9rdigiQXV0by1LTWVhbnMgc2VsZWN0aW9uIiwgeyJrIjogYmVzdF9rLCAic2lsaG91ZXR0ZSI6IHJvdW5kKGZsb2F0KGJlc3RfcyksNCl9KQoKCmNsYXNzIE1haW5XaW5kb3coUU1haW5XaW5kb3cpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYuc2V0V2luZG93VGl0bGUoV0lORE9XX1RJVExFKTsgc2VsZi5yZXNpemUoMTQyMCwgOTgwKQogICAgICAgIHNlbGYuZGY9Tm9uZTsgc2VsZi5tb2RlbD1QYW5kYXNNb2RlbChwZC5EYXRhRnJhbWUoKSkKICAgICAgICBzZWxmLnRhYnMgPSBRVGFiV2lkZ2V0KCk7IHNlbGYuc2V0Q2VudHJhbFdpZGdldChzZWxmLnRhYnMpCiAgICAgICAgc2VsZi5fYnVpbGRfd2VsY29tZV90YWIoKTsgc2VsZi5fYnVpbGRfZGF0YV90YWIoKTsgc2VsZi5fYnVpbGRfYW5hbHl6ZV90YWIoKTsgc2VsZi5fYnVpbGRfcmVzdWx0c190YWIoKTsgc2VsZi5fYnVpbGRfaGVscF90YWIoKTsgc2VsZi50YWJzLnNldEN1cnJlbnRJbmRleCgwKQogICAgICAgIHNlbGYucmVwb3J0ID0gUmVwb3J0QnVpbGRlcigpOyBzZWxmLnN0YXR1c0JhcigpLnNob3dNZXNzYWdlKCJSZWFkeS4gTG9hZCBDU1YvWExTWCB0byBiZWdpbi4iKTsgc2VsZi5zZXRBY2NlcHREcm9wcyhUcnVlKQoKICAgIAogICAgCiAgICBkZWYgX2J1aWxkX3dlbGNvbWVfdGFiKHNlbGYpOgogICAgICAgIHRhYiA9IFFXaWRnZXQoKTsgbGF5b3V0ID0gUVZCb3hMYXlvdXQodGFiKQoKICAgICAgICAKICAgICAgICBzZWxmLnJlYWRtZV92aWV3ID0gUVRleHRCcm93c2VyKCkKICAgICAgICBzZWxmLnJlYWRtZV92aWV3LnNldE9wZW5FeHRlcm5hbExpbmtzKFRydWUpCiAgICAgICAgcmVhZG1lX2h0bWwgPSBOb25lCiAgICAgICAgCiAgICAgICAgcmRfY2FuZGlkYXRlcyA9IFtvcy5wYXRoLmpvaW4oX2FwcF9kaXIoKSwgIlJFQURNRS5tZCIpLCBvcy5wYXRoLmpvaW4ob3MuZ2V0Y3dkKCksICJSRUFETUUubWQiKV0KICAgICAgICBmb3IgcCBpbiByZF9jYW5kaWRhdGVzOgogICAgICAgICAgICBpZiBvcy5wYXRoLmV4aXN0cyhwKToKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICB3aXRoIG9wZW4ocCwgInIiLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICAgICAgICAgICAgICB0eHQgPSBmLnJlYWQoKQogICAgICAgICAgICAgICAgICAgIHJlYWRtZV9odG1sID0gIjxoMj5SRUFETUU8L2gyPjxwcmUgc3R5bGU9J3doaXRlLXNwYWNlOnByZS13cmFwOyc+IiArIGh0bWwuZXNjYXBlKHR4dCkgKyAiPC9wcmU+IgogICAgICAgICAgICAgICAgICAgIGJyZWFrCiAgICAgICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOiBwYXNzCiAgICAgICAgaWYgbm90IHJlYWRtZV9odG1sOgogICAgICAgICAgICByZWFkbWVfaHRtbCA9ICI8aDI+UkVBRE1FPC9oMj48cD5SRUFETUUubWQgbm90IGZvdW5kIG5leHQgdG8gdGhlIGFwcGxpY2F0aW9uLjwvcD4iCiAgICAgICAgc2VsZi5yZWFkbWVfdmlldy5zZXRIdG1sKHJlYWRtZV9odG1sKQoKICAgICAgICAKICAgICAgICBzZWxmLnJlcXNfdmlldyA9IFFUZXh0QnJvd3NlcigpCiAgICAgICAgcmVxc19odG1sID0gWyI8aDI+UmVxdWlyZW1lbnRzIChzdGF0dXMgb24gdGhpcyBzeXN0ZW0pPC9oMj48dGFibGU+PHRyPjx0aD5QYWNrYWdlPC90aD48dGg+UmVxdWlyZWQ8L3RoPjx0aD5JbnN0YWxsZWQ8L3RoPjwvdHI+Il0KICAgICAgICByZXFfZmlsZSA9IE5vbmUKICAgICAgICBycV9jYW5kaWRhdGVzID0gW29zLnBhdGguam9pbihfYXBwX2RpcigpLCAicmVxdWlyZW1lbnRzLnR4dCIpLCBvcy5wYXRoLmpvaW4ob3MuZ2V0Y3dkKCksICJyZXF1aXJlbWVudHMudHh0IildCiAgICAgICAgZm9yIHAgaW4gcnFfY2FuZGlkYXRlczoKICAgICAgICAgICAgaWYgb3MucGF0aC5leGlzdHMocCk6CiAgICAgICAgICAgICAgICByZXFfZmlsZSA9IHA7IGJyZWFrCiAgICAgICAgcGtncyA9IFtdCiAgICAgICAgaWYgcmVxX2ZpbGU6CiAgICAgICAgICAgIHdpdGggb3BlbihyZXFfZmlsZSwgInIiLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICAgICAgZm9yIGxpbmUgaW4gZjoKICAgICAgICAgICAgICAgICAgICBsaW5lPWxpbmUuc3RyaXAoKQogICAgICAgICAgICAgICAgICAgIGlmIG5vdCBsaW5lIG9yIGxpbmUuc3RhcnRzd2l0aCgiIyIpOiBjb250aW51ZQogICAgICAgICAgICAgICAgICAgIHJlcSA9IGxpbmUKICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICBuYW1lID0gcmUuc3BsaXQociJbPD49IV0iLCBsaW5lLCBtYXhzcGxpdD0xKVswXS5zdHJpcCgpCiAgICAgICAgICAgICAgICAgICAgaWYgbm90IG5hbWU6IG5hbWUgPSBsaW5lLnN0cmlwKCkKICAgICAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgICAgIHZlciA9IGlsbS52ZXJzaW9uKG5hbWUpCiAgICAgICAgICAgICAgICAgICAgICAgIHN0YXR1cyA9IHZlcgogICAgICAgICAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICAgICAgICAgIHN0YXR1cyA9ICI8c3BhbiBzdHlsZT0nY29sb3I6I2I5MWMxYzsnPk1pc3Npbmc8L3NwYW4+IgogICAgICAgICAgICAgICAgICAgICAgICB2ZXIgPSAi4oCUIgogICAgICAgICAgICAgICAgICAgIHJlcXNfaHRtbC5hcHBlbmQoZiI8dHI+PHRkPntodG1sLmVzY2FwZShuYW1lKX08L3RkPjx0ZD57aHRtbC5lc2NhcGUocmVxKX08L3RkPjx0ZD57c3RhdHVzfTwvdGQ+PC90cj4iKQogICAgICAgICAgICAgICAgICAgIHBrZ3MuYXBwZW5kKG5hbWUpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcmVxc19odG1sLmFwcGVuZCgiPHRyPjx0ZCBjb2xzcGFuPSczJz5yZXF1aXJlbWVudHMudHh0IG5vdCBmb3VuZC48L3RkPjwvdHI+IikKICAgICAgICByZXFzX2h0bWwuYXBwZW5kKCI8L3RhYmxlPiIpCiAgICAgICAgc2VsZi5yZXFzX3ZpZXcuc2V0SHRtbCgiIi5qb2luKHJlcXNfaHRtbCkpCgogICAgICAgIAogICAgICAgIHNlbGYuc2FtcGxlc19saXN0ID0gUUxpc3RXaWRnZXQoKQogICAgICAgIHNhbXBfZGlyID0gTm9uZQogICAgICAgIGZvciBwIGluIFtvcy5wYXRoLmpvaW4oX2FwcF9kaXIoKSwgInNhbXBsZXMiKSwgb3MucGF0aC5qb2luKG9zLmdldGN3ZCgpLCAic2FtcGxlcyIpXToKICAgICAgICAgICAgaWYgb3MucGF0aC5pc2RpcihwKToKICAgICAgICAgICAgICAgIHNhbXBfZGlyID0gcDsgYnJlYWsKICAgICAgICBzZWxmLl9zYW1wbGVzX2RpciA9IHNhbXBfZGlyCiAgICAgICAgaWYgc2FtcF9kaXI6CiAgICAgICAgICAgIGZvciBmbiBpbiBzb3J0ZWQob3MubGlzdGRpcihzYW1wX2RpcikpOgogICAgICAgICAgICAgICAgaWYgZm4ubG93ZXIoKS5lbmRzd2l0aCgoIi5jc3YiLCIueGxzeCIsIi54bHMiKSk6CiAgICAgICAgICAgICAgICAgICAgaXRlbSA9IFFMaXN0V2lkZ2V0SXRlbShmbikKICAgICAgICAgICAgICAgICAgICBzZWxmLnNhbXBsZXNfbGlzdC5hZGRJdGVtKGl0ZW0pCgogICAgICAgIGJ0bnMgPSBRSEJveExheW91dCgpCiAgICAgICAgc2VsZi5idG5fbG9hZF9zYW1wbGUgPSBRUHVzaEJ1dHRvbigiT3BlbiBzZWxlY3RlZCBzYW1wbGUg4oaSIERhdGEgdGFiIikKICAgICAgICBzZWxmLmJ0bl9sb2FkX3NhbXBsZS5jbGlja2VkLmNvbm5lY3Qoc2VsZi5fb3Blbl9zZWxlY3RlZF9zYW1wbGUpCiAgICAgICAgYnRucy5hZGRXaWRnZXQoc2VsZi5idG5fbG9hZF9zYW1wbGUpOyBidG5zLmFkZFN0cmV0Y2goMSkKCiAgICAgICAgCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChRTGFiZWwoIjxiPldlbGNvbWUgdG8gU3RhdGlseXRpY3MgU3R1ZGlvIHYxLjA8L2I+IikpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnJlYWRtZV92aWV3KQogICAgICAgIGxheW91dC5hZGRXaWRnZXQoc2VsZi5yZXFzX3ZpZXcpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChRTGFiZWwoIjxiPlNhbXBsZSBkYXRhc2V0czwvYj4gKGRvdWJsZS1jbGljayBvciBzZWxlY3QgYW5kIGNsaWNrIE9wZW4pOiIpKQogICAgICAgIGxheW91dC5hZGRXaWRnZXQoc2VsZi5zYW1wbGVzX2xpc3QpCiAgICAgICAgbGF5b3V0LmFkZExheW91dChidG5zKQoKICAgICAgICAKICAgICAgICBzZWxmLnNhbXBsZXNfbGlzdC5pdGVtRG91YmxlQ2xpY2tlZC5jb25uZWN0KHNlbGYuX29wZW5fc2VsZWN0ZWRfc2FtcGxlKQoKICAgICAgICBzZWxmLnRhYnMuYWRkVGFiKHRhYiwgIldlbGNvbWUgLyBSZXNvdXJjZXMiKQoKICAgIGRlZiBfb3Blbl9zZWxlY3RlZF9zYW1wbGUoc2VsZik6CiAgICAgICAgaXRlbSA9IHNlbGYuc2FtcGxlc19saXN0LmN1cnJlbnRJdGVtKCkKICAgICAgICBpZiBub3QgaXRlbSBvciBub3Qgc2VsZi5fc2FtcGxlc19kaXI6CiAgICAgICAgICAgIFFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJTYW1wbGVzIiwgIk5vIHNhbXBsZSBzZWxlY3RlZC4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKHNlbGYuX3NhbXBsZXNfZGlyLCBpdGVtLnRleHQoKSkKICAgICAgICB0cnk6CiAgICAgICAgICAgIGlmIHBhdGgubG93ZXIoKS5lbmRzd2l0aCgoIi54bHN4IiwiLnhscyIpKToKICAgICAgICAgICAgICAgIGJvb2sgPSBwZC5yZWFkX2V4Y2VsKHBhdGgsIHNoZWV0X25hbWU9Tm9uZSkKICAgICAgICAgICAgICAgIGRmID0gYm9vay5nZXQoIkRhdGEiLCBuZXh0KGl0ZXIoYm9vay52YWx1ZXMoKSkpKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgZGYgPSBwZC5yZWFkX2NzdihwYXRoKQogICAgICAgICAgICBzZWxmLnNldF9kYXRhZnJhbWUoZGYpCiAgICAgICAgICAgIHNlbGYudGFicy5zZXRDdXJyZW50SW5kZXgoMSkgICMgc3dpdGNoIHRvIERhdGEgdGFiCiAgICAgICAgICAgIFFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJTYW1wbGUgbG9hZGVkIiwgZiJMb2FkZWQ6IHtvcy5wYXRoLmJhc2VuYW1lKHBhdGgpfSIpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAiU2FtcGxlIGVycm9yIiwgc3RyKGUpKQoKICAgIGRlZiBfYnVpbGRfZGF0YV90YWIoc2VsZik6CiAgICAgICAgdGFiID0gUVdpZGdldCgpOyBsYXkgPSBRVkJveExheW91dCh0YWIpCiAgICAgICAgc2VsZi50YWJsZSA9IFFUYWJsZVZpZXcoKTsgc2VsZi50YWJsZS5zZXRNb2RlbChzZWxmLm1vZGVsKQogICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbFNjcm9sbE1vZGUoUUFic3RyYWN0SXRlbVZpZXcuU2Nyb2xsUGVyUGl4ZWwpCiAgICAgICAgc2VsZi50YWJsZS5zZXRWZXJ0aWNhbFNjcm9sbE1vZGUoUUFic3RyYWN0SXRlbVZpZXcuU2Nyb2xsUGVyUGl4ZWwpCiAgICAgICAgc2VsZi50YWJsZS5zZXRIb3Jpem9udGFsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHNlbGYudGFibGUuc2V0VmVydGljYWxTY3JvbGxCYXJQb2xpY3koUXQuU2Nyb2xsQmFyQXNOZWVkZWQpCiAgICAgICAgc2VsZi50YWJsZS5zZXRXb3JkV3JhcChGYWxzZSkKICAgICAgICB0b3AgPSBRSEJveExheW91dCgpCiAgICAgICAgc2VsZi5idG5fb3BlbiA9IFFQdXNoQnV0dG9uKCJPcGVuIENTVi9YTFNY4oCmIik7IHNlbGYuYnRuX29wZW4uY2xpY2tlZC5jb25uZWN0KHNlbGYub3Blbl9maWxlKQogICAgICAgIHNlbGYubGJsX3NoYXBlID0gUUxhYmVsKCJObyBkYXRhIGxvYWRlZCIpCiAgICAgICAgdG9wLmFkZFdpZGdldChzZWxmLmJ0bl9vcGVuKTsgdG9wLmFkZFN0cmV0Y2goMSk7IHRvcC5hZGRXaWRnZXQoc2VsZi5sYmxfc2hhcGUpCiAgICAgICAgbGF5LmFkZExheW91dCh0b3ApOyBsYXkuYWRkV2lkZ2V0KHNlbGYudGFibGUpCiAgICAgICAgc2VsZi50YWJzLmFkZFRhYih0YWIsICJEYXRhIikKCiAgICBkZWYgX2J1aWxkX2FuYWx5emVfdGFiKHNlbGYpOgogICAgICAgIHRhYiA9IFFXaWRnZXQoKTsgbGF5b3V0ID0gUUhCb3hMYXlvdXQodGFiKQoKICAgICAgICBjb250cm9sc19ncm91cCA9IFFHcm91cEJveCgiUHJvY2VkdXJlcyAmIE9wdGlvbnMiKQogICAgICAgIGZvcm0gPSBRRm9ybUxheW91dChjb250cm9sc19ncm91cCkKICAgICAgICBmb3JtLnNldExhYmVsQWxpZ25tZW50KFF0LkFsaWduTGVmdCkKICAgICAgICBmb3JtLnNldEZpZWxkR3Jvd3RoUG9saWN5KFFGb3JtTGF5b3V0LkFsbE5vbkZpeGVkRmllbGRzR3JvdykKCiAgICAgICAgc2VsZi5jbWJfcHJvYyA9IFFDb21ib0JveCgpOyBzZWxmLmNtYl9wcm9jLmFkZEl0ZW1zKFsKICAgICAgICAgICAgIkRlc2NyaWJlIGRhdGEiLCJDcm9zc3RhYiAoY2hpLXNxdWFyZS9GaXNoZXIpIiwKICAgICAgICAgICAgInQtdGVzdCAob25lLXNhbXBsZSkiLCJ0LXRlc3QgKGluZGVwZW5kZW50KSIsInQtdGVzdCAocGFpcmVkKSIsCiAgICAgICAgICAgICJBTk9WQSAob25lLXdheSkiLCJSZXBlYXRlZC1tZWFzdXJlcyBBTk9WQSIsIkFOQ09WQSIsIlR1a2V5IEhTRCIsCiAgICAgICAgICAgICJNYW5u4oCTV2hpdG5leSBVIiwiV2lsY294b24gc2lnbmVkLXJhbmsiLCJLcnVza2Fs4oCTV2FsbGlzIiwiRnJpZWRtYW4iLAogICAgICAgICAgICAiQ29ycmVsYXRpb24gKFBlYXJzb24pIiwiQ29ycmVsYXRpb24gKFNwZWFybWFuKSIsIkNvcnJlbGF0aW9uIChLZW5kYWxsKSIsCiAgICAgICAgICAgICJPTFMgcmVncmVzc2lvbiIsIkxvZ2lzdGljIHJlZ3Jlc3Npb24iLCJNdWx0aW5vbWlhbCBsb2dpdCIsIk9yZGluYWwgcmVncmVzc2lvbiAobG9naXQpIiwKICAgICAgICAgICAgIkdMTSAoUG9pc3NvbikiLCJHTE0gKE5lZ2F0aXZlIGJpbm9taWFsKSIsIkdMTSAoR2FtbWEpIiwKICAgICAgICAgICAgIkxpbmVhciBtaXhlZCBtb2RlbCIsCiAgICAgICAgICAgICJDcm9uYmFjaCBhbHBoYSIsIkNvaGVuIGthcHBhIiwiV2VpZ2h0ZWQga2FwcGEgKGxpbmVhcikiLCJXZWlnaHRlZCBrYXBwYSAocXVhZHJhdGljKSIsCiAgICAgICAgICAgICJGbGVpc3MnIGthcHBhIChtdWx0aS1yYXRlcikiLCJJQ0MiLAogICAgICAgICAgICAiS3JpcHBlbmRvcmZm4oCZcyBhbHBoYSAobm9taW5hbCkiLCJLcmlwcGVuZG9yZmbigJlzIGFscGhhIChvcmRpbmFsKSIsIktyaXBwZW5kb3JmZuKAmXMgYWxwaGEgKGludGVydmFsKSIsCiAgICAgICAgICAgICJTdXJ2aXZhbCAoS00gKyBDb3ggKyBXZWlidWxsIEFGVCkiLAogICAgICAgICAgICAiUk9DIC8gUFIgLyBCcmllciIsCiAgICAgICAgICAgICJBUklNQSBmb3JlY2FzdCIsIkVUUyBmb3JlY2FzdCIsCiAgICAgICAgICAgICJNZXRhLWFuYWx5c2lzIChETCByYW5kb20tZWZmZWN0cykiLAogICAgICAgICAgICAiSVBUVyAoQVRFKSIsIkRpZmZlcmVuY2UtaW4tRGlmZmVyZW5jZXMiLAogICAgICAgICAgICAiUENBIiwiRUZBICh2YXJpbWF4KSIsIkNvcnJlc3BvbmRlbmNlIGFuYWx5c2lzIiwKICAgICAgICAgICAgIkxEQSAoY2xhc3NpZmljYXRpb24pIiwiRGVjaXNpb24gdHJlZSAoY2xhc3NpZmljYXRpb24pIiwiQ0hBSUQtbGlrZSB0cmVlIiwiUmFuZG9tIGZvcmVzdCAoY2xhc3NpZmljYXRpb24pIiwKICAgICAgICAgICAgIkstTWVhbnMiLCJBZ2dsb21lcmF0aXZlIGNsdXN0ZXJpbmciLCJBdXRvLUtNZWFucyAoVHdvU3RlcC1pbnNwaXJlZCkiLAogICAgICAgICAgICAiTUFOT1ZBIC8gTUFOQ09WQSIKICAgICAgICBdKTsgZm9ybS5hZGRSb3coIlByb2NlZHVyZSIsIHNlbGYuY21iX3Byb2MpCgogICAgICAgIHNlbGYuY21iX3kgPSBRQ29tYm9Cb3goKTsgZm9ybS5hZGRSb3coIlkgLyBPdXRjb21lIiwgc2VsZi5jbWJfeSkKICAgICAgICBzZWxmLmNtYl94ID0gUUNvbWJvQm94KCk7IGZvcm0uYWRkUm93KCJYIC8gU2Vjb25kIFZhciIsIHNlbGYuY21iX3gpCiAgICAgICAgc2VsZi5jbWJfZ3JvdXAgPSBRQ29tYm9Cb3goKTsgZm9ybS5hZGRSb3coIkdyb3VwIC8gRmFjdG9yIiwgc2VsZi5jbWJfZ3JvdXApCiAgICAgICAgc2VsZi5jbWJfZ3JvdXAyID0gUUNvbWJvQm94KCk7IGZvcm0uYWRkUm93KCJTZWNvbmQgR3JvdXAgKG9wdGlvbmFsKSIsIHNlbGYuY21iX2dyb3VwMikKICAgICAgICBzZWxmLnR4dF9wcmVkaWN0b3JzID0gUUxpbmVFZGl0KCk7IGZvcm0uYWRkUm93KCJQcmVkaWN0b3JzIChjb21tYS9zZW1pY29sb24vc3BhY2UsIHF1b3RlcyBPSykiLCBzZWxmLnR4dF9wcmVkaWN0b3JzKQogICAgICAgIHNlbGYuY21iX3RpbWUgPSBRQ29tYm9Cb3goKTsgZm9ybS5hZGRSb3coIlRpbWUgKHN1cnZpdmFsL1RTKSIsIHNlbGYuY21iX3RpbWUpCiAgICAgICAgc2VsZi5jbWJfZXZlbnQgPSBRQ29tYm9Cb3goKTsgZm9ybS5hZGRSb3coIkV2ZW50ICgwLzEpIiwgc2VsZi5jbWJfZXZlbnQpCiAgICAgICAgc2VsZi5jbWJfY2x1c3RlciA9IFFDb21ib0JveCgpOyBmb3JtLmFkZFJvdygiQ2x1c3RlciAoTE1NKSIsIHNlbGYuY21iX2NsdXN0ZXIpCiAgICAgICAgc2VsZi5jbWJfc3ViamVjdCA9IFFDb21ib0JveCgpOyBmb3JtLmFkZFJvdygiU3ViamVjdCBJRCAoUk0gQU5PVkEpIiwgc2VsZi5jbWJfc3ViamVjdCkKICAgICAgICBzZWxmLnR4dF93aXRoaW4gPSBRTGluZUVkaXQoKTsgZm9ybS5hZGRSb3coIldpdGhpbi1mYWN0b3IgbmFtZSAoUk0gQU5PVkEpIiwgc2VsZi50eHRfd2l0aGluKQogICAgICAgIHNlbGYudHh0X211bHRpX3kgPSBRTGluZUVkaXQoKTsgZm9ybS5hZGRSb3coIlkncyAoY29tbWEgZm9yIE1BTk9WQSkiLCBzZWxmLnR4dF9tdWx0aV95KQogICAgICAgIHNlbGYudHh0X2NvbnN0ID0gUUxpbmVFZGl0KCk7IGZvcm0uYWRkUm93KCJDb25zdGFudCAvIM68Iiwgc2VsZi50eHRfY29uc3QpCiAgICAgICAgc2VsZi50eHRfbGV2ZWxzID0gUUxpbmVFZGl0KCk7IGZvcm0uYWRkUm93KCJHcm91cCBsZXZlbHMiLCBzZWxmLnR4dF9sZXZlbHMpCiAgICAgICAgc2VsZi5zcG5fc3RlcHMgPSBRU3BpbkJveCgpOyBzZWxmLnNwbl9zdGVwcy5zZXRSYW5nZSgxLCA1MDApOyBzZWxmLnNwbl9zdGVwcy5zZXRWYWx1ZSgxMik7IGZvcm0uYWRkUm93KCJGb3JlY2FzdCBzdGVwcyIsIHNlbGYuc3BuX3N0ZXBzKQogICAgICAgIHNlbGYuc3BuX2sgPSBRU3BpbkJveCgpOyBzZWxmLnNwbl9rLnNldFJhbmdlKDIsIDIwKTsgc2VsZi5zcG5fay5zZXRWYWx1ZSgzKTsgZm9ybS5hZGRSb3coIkNvbXBvbmVudHMgLyBDbHVzdGVycyAoaykiLCBzZWxmLnNwbl9rKQogICAgICAgIHNlbGYuY2hrX2Zpc2hlciA9IFFDaGVja0JveCgiVXNlIEZpc2hlciBleGFjdCAoMngyKSIpOyBmb3JtLmFkZFJvdyhzZWxmLmNoa19maXNoZXIpCiAgICAgICAgc2VsZi5idG5fcnVuID0gUVB1c2hCdXR0b24oIlJ1biIpOyBzZWxmLmJ0bl9ydW4uc2V0U2l6ZVBvbGljeShRU2l6ZVBvbGljeS5QcmVmZXJyZWQsIFFTaXplUG9saWN5LkZpeGVkKTsgc2VsZi5idG5fcnVuLmNsaWNrZWQuY29ubmVjdChzZWxmLnJ1bik7IGZvcm0uYWRkUm93KHNlbGYuYnRuX3J1bikKCiAgICAgICAgbGVmdF9jb250YWluZXIgPSBRV2lkZ2V0KCk7IGxlZnRfdiA9IFFWQm94TGF5b3V0KGxlZnRfY29udGFpbmVyKTsgbGVmdF92LmFkZFdpZGdldChjb250cm9sc19ncm91cCk7IGxlZnRfdi5hZGRTdHJldGNoKDEpCiAgICAgICAgc2Nyb2xsX2xlZnQgPSBRU2Nyb2xsQXJlYSgpOyBzY3JvbGxfbGVmdC5zZXRXaWRnZXRSZXNpemFibGUoVHJ1ZSkKICAgICAgICBzY3JvbGxfbGVmdC5zZXRIb3Jpem9udGFsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHNjcm9sbF9sZWZ0LnNldFZlcnRpY2FsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHNjcm9sbF9sZWZ0LnNldFdpZGdldChsZWZ0X2NvbnRhaW5lcikKICAgICAgICBzY3JvbGxfbGVmdC5zZXRNaW5pbXVtV2lkdGgoNDMwKQoKICAgICAgICByaWdodCA9IFFXaWRnZXQoKTsgcmxheSA9IFFWQm94TGF5b3V0KHJpZ2h0KQogICAgICAgIHNlbGYudHh0X3JlcG9ydCA9IFFUZXh0RWRpdCgpOyBzZWxmLnR4dF9yZXBvcnQuc2V0UmVhZE9ubHkoVHJ1ZSkKICAgICAgICBzZWxmLnR4dF9yZXBvcnQuc2V0TGluZVdyYXBNb2RlKFFUZXh0RWRpdC5Ob1dyYXApCiAgICAgICAgc2VsZi50eHRfcmVwb3J0LnNldFZlcnRpY2FsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHNlbGYudHh0X3JlcG9ydC5zZXRIb3Jpem9udGFsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHJsYXkuYWRkV2lkZ2V0KHNlbGYudHh0X3JlcG9ydCkKCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzY3JvbGxfbGVmdCwgMCkKICAgICAgICBsYXlvdXQuYWRkV2lkZ2V0KHJpZ2h0LCAxKQogICAgICAgIHNlbGYudGFicy5hZGRUYWIodGFiLCAiQW5hbHl6ZSIpCgogICAgCiAgICBkZWYgX2J1aWxkX2hlbHBfdGFiKHNlbGYpOgogICAgICAgIHRhYiA9IFFXaWRnZXQoKTsgbGF5b3V0ID0gUVZCb3hMYXlvdXQodGFiKQogICAgICAgIHNlbGYuaGVscF92aWV3ID0gUVRleHRCcm93c2VyKCkKICAgICAgICBzZWxmLmhlbHBfdmlldy5zZXRPcGVuRXh0ZXJuYWxMaW5rcyhUcnVlKQoKICAgICAgICAKICAgICAgICBhcHBfZGlyID0gTm9uZQogICAgICAgIHRyeToKICAgICAgICAgICAgYXBwX2RpciA9IG9zLnBhdGguZGlybmFtZShvcy5wYXRoLmFic3BhdGgoX19maWxlX18pKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIGFwcF9kaXIgPSBvcy5wYXRoLmRpcm5hbWUob3MucGF0aC5hYnNwYXRoKHN5cy5hcmd2WzBdKSkKICAgICAgICBpZiBnZXRhdHRyKHN5cywgImZyb3plbiIsIEZhbHNlKToKICAgICAgICAgICAgYXBwX2RpciA9IGdldGF0dHIoc3lzLCAiX01FSVBBU1MiLCBhcHBfZGlyKQoKICAgICAgICBjYW5kaWRhdGVzID0gWwogICAgICAgICAgICBvcy5wYXRoLmpvaW4oYXBwX2RpciwgIlVTRVJfTUFOVUFMLmh0bWwiKSwKICAgICAgICAgICAgb3MucGF0aC5qb2luKG9zLmdldGN3ZCgpLCAiVVNFUl9NQU5VQUwuaHRtbCIpCiAgICAgICAgXQogICAgICAgIG1hbnVhbF9wYXRoID0gbmV4dCgocCBmb3IgcCBpbiBjYW5kaWRhdGVzIGlmIG9zLnBhdGguZXhpc3RzKHApKSwgTm9uZSkKCiAgICAgICAgaWYgbWFudWFsX3BhdGg6CiAgICAgICAgICAgIHNlbGYuaGVscF92aWV3LnNldFNvdXJjZShRdENvcmUuUVVybC5mcm9tTG9jYWxGaWxlKG1hbnVhbF9wYXRoKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBzZWxmLmhlbHBfdmlldy5zZXRIdG1sKGYiPGgyPntBUFBfTkFNRX08L2gyPjxwPntDT1BZUklHSFR9PC9wPjxwPjxiPlVzZXIgTWFudWFsIG5vdCBmb3VuZC48L2I+IFBsYWNlIDxjb2RlPlVTRVJfTUFOVUFMLmh0bWw8L2NvZGU+IGluIHRoZSBzYW1lIGZvbGRlciBhcyB0aGUgYXBwbGljYXRpb24uPC9wPiIpCgogICAgICAgIGxheW91dC5hZGRXaWRnZXQoc2VsZi5oZWxwX3ZpZXcpCiAgICAgICAgc2VsZi50YWJzLmFkZFRhYih0YWIsICJIZWxwIC8gTWFudWFsIikKCiAgICBkZWYgX2J1aWxkX3Jlc3VsdHNfdGFiKHNlbGYpOgogICAgICAgIHRhYiA9IFFXaWRnZXQoKTsgbGF5b3V0ID0gUVZCb3hMYXlvdXQodGFiKQogICAgICAgIHRvcCA9IFFIQm94TGF5b3V0KCkKICAgICAgICBzZWxmLmJ0bl9leHBvcnRfaHRtbCA9IFFQdXNoQnV0dG9uKCJFeHBvcnQgSFRNTOKApiIpOyBzZWxmLmJ0bl9leHBvcnRfaHRtbC5jbGlja2VkLmNvbm5lY3Qoc2VsZi5leHBvcnRfaHRtbCkKICAgICAgICBzZWxmLmJ0bl9leHBvcnRfZG9jeCA9IFFQdXNoQnV0dG9uKCJFeHBvcnQgRE9DWOKApiIpOyBzZWxmLmJ0bl9leHBvcnRfZG9jeC5jbGlja2VkLmNvbm5lY3Qoc2VsZi5leHBvcnRfZG9jeCkKICAgICAgICB0b3AuYWRkV2lkZ2V0KHNlbGYuYnRuX2V4cG9ydF9odG1sKTsgdG9wLmFkZFdpZGdldChzZWxmLmJ0bl9leHBvcnRfZG9jeCk7IHRvcC5hZGRTdHJldGNoKDEpCiAgICAgICAgc2VsZi50eHRfZmluYWwgPSBRVGV4dEVkaXQoKTsgc2VsZi50eHRfZmluYWwuc2V0UmVhZE9ubHkoVHJ1ZSk7IHNlbGYudHh0X2ZpbmFsLnNldExpbmVXcmFwTW9kZShRVGV4dEVkaXQuTm9XcmFwKQogICAgICAgIHNlbGYudHh0X2ZpbmFsLnNldFZlcnRpY2FsU2Nyb2xsQmFyUG9saWN5KFF0LlNjcm9sbEJhckFzTmVlZGVkKQogICAgICAgIHNlbGYudHh0X2ZpbmFsLnNldEhvcml6b250YWxTY3JvbGxCYXJQb2xpY3koUXQuU2Nyb2xsQmFyQXNOZWVkZWQpCiAgICAgICAgbGF5b3V0LmFkZExheW91dCh0b3ApOyBsYXlvdXQuYWRkV2lkZ2V0KHNlbGYudHh0X2ZpbmFsKQogICAgICAgIHNlbGYudGFicy5hZGRUYWIodGFiLCAiUmVzdWx0cyAmIEV4cG9ydCIpCgogICAgCiAgICBkZWYgb3Blbl9maWxlKHNlbGYpOgogICAgICAgIHBhdGgsIF8gPSBRRmlsZURpYWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgIk9wZW4gRGF0YSIsIG9zLmdldGN3ZCgpLCAiRGF0YSAoKi5jc3YgKi54bHN4ICoueGxzKSIpCiAgICAgICAgaWYgbm90IHBhdGg6IHJldHVybgogICAgICAgIHRyeToKICAgICAgICAgICAgaWYgcGF0aC5sb3dlcigpLmVuZHN3aXRoKCgiLnhsc3giLCIueGxzIikpOgogICAgICAgICAgICAgICAgYm9vayA9IHBkLnJlYWRfZXhjZWwocGF0aCwgc2hlZXRfbmFtZT1Ob25lKQogICAgICAgICAgICAgICAgZGYgPSBib29rLmdldCgiRGF0YSIsIG5leHQoaXRlcihib29rLnZhbHVlcygpKSkpCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBkZiA9IHBkLnJlYWRfY3N2KHBhdGgpCiAgICAgICAgICAgIHNlbGYuc2V0X2RhdGFmcmFtZShkZikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsIkZpbGUgZXJyb3IiLHN0cihlKSkKCiAgICBkZWYgc2V0X2RhdGFmcmFtZShzZWxmLCBkZjogcGQuRGF0YUZyYW1lKToKICAgICAgICBzZWxmLmRmPWRmOyBzZWxmLm1vZGVsLnNldERhdGFGcmFtZShkZik7IHNlbGYubGJsX3NoYXBlLnNldFRleHQoZiJSb3dzOiB7ZGYuc2hhcGVbMF19IHwgQ29sczoge2RmLnNoYXBlWzFdfSIpCiAgICAgICAgY29scyA9IFsi4oCUIl0gKyBsaXN0KGRmLmNvbHVtbnMpCiAgICAgICAgY29tYm9zID0gW3NlbGYuY21iX3ksIHNlbGYuY21iX3gsIHNlbGYuY21iX2dyb3VwLCBzZWxmLmNtYl9ncm91cDIsIHNlbGYuY21iX3RpbWUsIHNlbGYuY21iX2V2ZW50LCBzZWxmLmNtYl9jbHVzdGVyLCBzZWxmLmNtYl9zdWJqZWN0XQogICAgICAgIGZvciBjbWIgaW4gY29tYm9zOgogICAgICAgICAgICBjbWIuY2xlYXIoKTsgY21iLmFkZEl0ZW1zKGNvbHMpCiAgICAgICAgc2VsZi5yZXBvcnQgPSBSZXBvcnRCdWlsZGVyKCk7IHNlbGYucmVwb3J0LmFkZF9pbmZvKCJEYXRhIGxvYWRlZCIsIGYiU2hhcGU6IHtkZi5zaGFwZX0iKQogICAgICAgIHNlbGYuX3JlZnJlc2goKQoKICAgIAogICAgZGVmIF9zcGxpdF9zaW1wbGUoc2VsZiwgdHh0OiBzdHIpOgogICAgICAgIHJldHVybiBbdC5zdHJpcCgpIGZvciB0IGluIHJlLnNwbGl0KHIiW1xccyw7XSsiLCB0eHQgb3IgIiIpIGlmIHQuc3RyaXAoKV0KCiAgICBkZWYgX3BhcnNlX2NvbHMoc2VsZiwgdHh0OiBzdHIpOgogICAgICAgICIiIlBhcnNlIFByZWRpY3RvcnMgYWxsb3dpbmcgY29tbWFzL3NlbWljb2xvbnMvc3BhY2VzIGFuZCBxdW90ZWQgbmFtZXMuIiIiCiAgICAgICAgaWYgbm90IHR4dCBvciBub3QgdHh0LnN0cmlwKCk6CiAgICAgICAgICAgIHJldHVybiBbXQogICAgICAgIHMgPSB0eHQucmVwbGFjZSgiOyIsICIsIikuc3RyaXAoKQogICAgICAgIGNvbHMgPSBbXQogICAgICAgIHRyeToKICAgICAgICAgICAgY29scyA9IG5leHQoY3N2LnJlYWRlcihbc10sIHNraXBpbml0aWFsc3BhY2U9VHJ1ZSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgY29scyA9IFtdCiAgICAgICAgaWYgbGVuKGNvbHMpIDw9IDEgYW5kICgiLCIgbm90IGluIHMpOgogICAgICAgICAgICBjb2xzID0gW3QgZm9yIHQgaW4gcmUuc3BsaXQociJcXHMrIiwgcykgaWYgdF0KICAgICAgICByZXR1cm4gW2Muc3RyaXAoKS5zdHJpcCgnIicpLnN0cmlwKCInIikgZm9yIGMgaW4gY29scyBpZiBjLnN0cmlwKCkuc3RyaXAoJyInKS5zdHJpcCgiJyIpXQoKICAgIGRlZiBfdmFsKHNlbGYsIGNvbWJvOiBRQ29tYm9Cb3gpOgogICAgICAgIHJldHVybiBOb25lIGlmIGNvbWJvLmN1cnJlbnRUZXh0KCk9PSLigJQiIGVsc2UgY29tYm8uY3VycmVudFRleHQoKQoKICAgIAogICAgZGVmIHJ1bihzZWxmKToKICAgICAgICBpZiBzZWxmLmRmIGlzIE5vbmU6IFFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwiTm8gZGF0YSIsIkxvYWQgYSBkYXRhc2V0IGZpcnN0LiIpOyByZXR1cm4KICAgICAgICBFID0gRW5naW5lKHNlbGYuZGYpOyBSID0gUmVwb3J0QnVpbGRlcigpCiAgICAgICAgcCA9IHNlbGYuY21iX3Byb2MuY3VycmVudFRleHQoKQogICAgICAgIHZhbCA9IHNlbGYuX3ZhbAoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGlmIHA9PSJEZXNjcmliZSBkYXRhIjoKICAgICAgICAgICAgICAgIFIuYWRkX3RhYmxlKEUuZGVzY3JpYmUoKS5yb3VuZCg0KSwgIkRlc2NyaXB0aXZlcyIpCgogICAgICAgICAgICBlbGlmIHA9PSJDcm9zc3RhYiAoY2hpLXNxdWFyZS9GaXNoZXIpIjoKICAgICAgICAgICAgICAgIGEgPSB2YWwoc2VsZi5jbWJfeSk7IGIgPSB2YWwoc2VsZi5jbWJfeCk7IAogICAgICAgICAgICAgICAgaWYgbm90IGEgb3Igbm90IGI6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgYm90aCBZIGFuZCBYIGZvciBjcm9zc3RhYi4iKQogICAgICAgICAgICAgICAgdGFiLCBpbmZvID0gRS5jcm9zc3RhYihhLGIsZmlzaGVyPXNlbGYuY2hrX2Zpc2hlci5pc0NoZWNrZWQoKSk7IFIuYWRkX3RhYmxlKHRhYiwiQ3Jvc3N0YWIiKTsgUi5hZGRfa3YoIkFzc29jaWF0aW9uIHRlc3QiLCBpbmZvKQoKICAgICAgICAgICAgZWxpZiBwPT0idC10ZXN0IChvbmUtc2FtcGxlKSI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpOiByYWlzZSBWYWx1ZUVycm9yKCJTZWxlY3QgWSBmb3Igb25lLXNhbXBsZSB0LiIpCiAgICAgICAgICAgICAgICBSLmFkZF9rdigiT25lLXNhbXBsZSB0IiwgRS50X29uZV9zYW1wbGUodmFsKHNlbGYuY21iX3kpLCBmbG9hdChzZWxmLnR4dF9jb25zdC50ZXh0KCkgb3IgMC4wKSkpCgogICAgICAgICAgICBlbGlmIHA9PSJ0LXRlc3QgKGluZGVwZW5kZW50KSI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpIG9yIG5vdCB2YWwoc2VsZi5jbWJfZ3JvdXApOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgYW5kIEdyb3VwIGZvciBpbmRlcGVuZGVudCB0LiIpCiAgICAgICAgICAgICAgICBsZXZlbHMgPSAoc2VsZi50eHRfbGV2ZWxzLnRleHQoKSBvciAiIikuc3BsaXQoIiwiKTsgCiAgICAgICAgICAgICAgICBpZiBsZW4obGV2ZWxzKTwyOiByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIHR3byBncm91cCBsZXZlbHMsIGUuZy4sIEEsQiIpCiAgICAgICAgICAgICAgICBSLmFkZF9rdigiSW5kZXBlbmRlbnQgdCIsIEUudF9pbmQodmFsKHNlbGYuY21iX3kpLCB2YWwoc2VsZi5jbWJfZ3JvdXApLCBsZXZlbHNbMF0uc3RyaXAoKSwgbGV2ZWxzWzFdLnN0cmlwKCksIGVxdWFsX3Zhcj1GYWxzZSkpCgogICAgICAgICAgICBlbGlmIHA9PSJ0LXRlc3QgKHBhaXJlZCkiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KSBvciBub3QgdmFsKHNlbGYuY21iX3gpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgYW5kIFggZm9yIHBhaXJlZCB0LiIpCiAgICAgICAgICAgICAgICBSLmFkZF9rdigiUGFpcmVkIHQiLCBFLnRfcGFpcmVkKHZhbChzZWxmLmNtYl95KSwgdmFsKHNlbGYuY21iX3gpKSkKCiAgICAgICAgICAgIGVsaWYgcD09IkFOT1ZBIChvbmUtd2F5KSI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpIG9yIG5vdCB2YWwoc2VsZi5jbWJfZ3JvdXApOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgYW5kIEdyb3VwIGZvciBBTk9WQS4iKQogICAgICAgICAgICAgICAgRixwdiA9IEUuYW5vdmFfb25ld2F5KHZhbChzZWxmLmNtYl95KSwgdmFsKHNlbGYuY21iX2dyb3VwKSk7IFIuYWRkX2t2KCJPbmUtd2F5IEFOT1ZBIiwgeyJGIjogcm91bmQoRiw0KSwicCI6IHJvdW5kKHB2LDYpfSkKCiAgICAgICAgICAgIGVsaWYgcD09IlJlcGVhdGVkLW1lYXN1cmVzIEFOT1ZBIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSkgb3Igbm90IHZhbChzZWxmLmNtYl9zdWJqZWN0KSBvciBub3Qgc2VsZi50eHRfd2l0aGluLnRleHQoKS5zdHJpcCgpOiByYWlzZSBWYWx1ZUVycm9yKCJZLCBTdWJqZWN0LCBhbmQgV2l0aGluIG5hbWUgcmVxdWlyZWQuIikKICAgICAgICAgICAgICAgIFIuYWRkX3RhYmxlKEUuYW5vdmFfcm0odmFsKHNlbGYuY21iX3kpLCB2YWwoc2VsZi5jbWJfc3ViamVjdCksIHNlbGYudHh0X3dpdGhpbi50ZXh0KCkuc3RyaXAoKSksICJSZXBlYXRlZC1tZWFzdXJlcyBBTk9WQSIpCgogICAgICAgICAgICBlbGlmIHA9PSJBTkNPVkEiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KSBvciBub3QgdmFsKHNlbGYuY21iX2dyb3VwKTogcmFpc2UgVmFsdWVFcnJvcigiUGljayBZIGFuZCBHcm91cCBmb3IgQU5DT1ZBLiIpCiAgICAgICAgICAgICAgICBtID0gRS5hbmNvdmEodmFsKHNlbGYuY21iX3kpLCB2YWwoc2VsZi5jbWJfZ3JvdXApLCBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpKTsgUi5hZGRfdGFibGUobS5zdW1tYXJ5MigpLnRhYmxlc1sxXS5yZXNldF9pbmRleCgpLCAiQU5DT1ZBIChPTFMgdGFibGUpIikKCiAgICAgICAgICAgIGVsaWYgcD09IlR1a2V5IEhTRCI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpIG9yIG5vdCB2YWwoc2VsZi5jbWJfZ3JvdXApOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgYW5kIEdyb3VwIGZvciBUdWtleS4iKQogICAgICAgICAgICAgICAgUi5hZGRfdGFibGUoRS50dWtleV9oc2QodmFsKHNlbGYuY21iX3kpLCB2YWwoc2VsZi5jbWJfZ3JvdXApKSwgIlR1a2V5IEhTRCBwYWlyd2lzZSIpCgogICAgICAgICAgICBlbGlmIHA9PSJNYW5u4oCTV2hpdG5leSBVIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSkgb3Igbm90IHZhbChzZWxmLmNtYl9ncm91cCk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBhbmQgR3JvdXAgZm9yIE1hbm7igJNXaGl0bmV5LiIpCiAgICAgICAgICAgICAgICBsZXZlbHM9KHNlbGYudHh0X2xldmVscy50ZXh0KCkgb3IgIiIpLnNwbGl0KCIsIik7IAogICAgICAgICAgICAgICAgaWYgbGVuKGxldmVscyk8MjogcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSB0d28gZ3JvdXAgbGV2ZWxzLCBlLmcuLCBBLEIiKQogICAgICAgICAgICAgICAgUi5hZGRfa3YoIk1hbm7igJNXaGl0bmV5IFUiLCBFLm1hbm53aGl0bmV5KHZhbChzZWxmLmNtYl95KSwgdmFsKHNlbGYuY21iX2dyb3VwKSwgbGV2ZWxzWzBdLnN0cmlwKCksIGxldmVsc1sxXS5zdHJpcCgpKSkKCiAgICAgICAgICAgIGVsaWYgcD09IldpbGNveG9uIHNpZ25lZC1yYW5rIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSkgb3Igbm90IHZhbChzZWxmLmNtYl94KTogcmFpc2UgVmFsdWVFcnJvcigiUGljayBZIGFuZCBYIGZvciBXaWxjb3hvbi4iKQogICAgICAgICAgICAgICAgUi5hZGRfa3YoIldpbGNveG9uIHNpZ25lZC1yYW5rIiwgRS53aWxjb3hvbih2YWwoc2VsZi5jbWJfeSksIHZhbChzZWxmLmNtYl94KSkpCgogICAgICAgICAgICBlbGlmIHA9PSJLcnVza2Fs4oCTV2FsbGlzIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSkgb3Igbm90IHZhbChzZWxmLmNtYl9ncm91cCk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBhbmQgR3JvdXAgZm9yIEtydXNrYWzigJNXYWxsaXMuIikKICAgICAgICAgICAgICAgIFIuYWRkX2t2KCJLcnVza2Fs4oCTV2FsbGlzIiwgRS5rcnVza2FsKHZhbChzZWxmLmNtYl95KSwgdmFsKHNlbGYuY21iX2dyb3VwKSkpCgogICAgICAgICAgICBlbGlmIHA9PSJGcmllZG1hbiI6CiAgICAgICAgICAgICAgICBjb2xzID0gc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKQogICAgICAgICAgICAgICAgaWYgbGVuKGNvbHMpPDM6IHJhaXNlIFZhbHVlRXJyb3IoIlByb3ZpZGUgMyBjb2x1bW5zIGZvciBGcmllZG1hbi4iKQogICAgICAgICAgICAgICAgUi5hZGRfa3YoIkZyaWVkbWFuIiwgRS5mcmllZG1hbihjb2xzKSkKCiAgICAgICAgICAgIGVsaWYgcD09IkNvcnJlbGF0aW9uIChQZWFyc29uKSI6CiAgICAgICAgICAgICAgICBjb2xzID0gc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKTsgCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scyk8MjogcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSBhdCBsZWFzdCAyIGNvbHVtbnMgZm9yIGNvcnJlbGF0aW9uLiIpCiAgICAgICAgICAgICAgICBSLmFkZF90YWJsZShFLmNvcnJlbGF0aW9ucyhjb2xzLCAncGVhcnNvbicpLnJvdW5kKDQpLCJQZWFyc29uIGNvcnJlbGF0aW9uIikKCiAgICAgICAgICAgIGVsaWYgcD09IkNvcnJlbGF0aW9uIChTcGVhcm1hbikiOgogICAgICAgICAgICAgICAgY29scyA9IHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSk7IAogICAgICAgICAgICAgICAgaWYgbGVuKGNvbHMpPDI6IHJhaXNlIFZhbHVlRXJyb3IoIlByb3ZpZGUgYXQgbGVhc3QgMiBjb2x1bW5zIGZvciBjb3JyZWxhdGlvbi4iKQogICAgICAgICAgICAgICAgUi5hZGRfdGFibGUoRS5jb3JyZWxhdGlvbnMoY29scywgJ3NwZWFybWFuJykucm91bmQoNCksIlNwZWFybWFuIGNvcnJlbGF0aW9uIikKCiAgICAgICAgICAgIGVsaWYgcD09IkNvcnJlbGF0aW9uIChLZW5kYWxsKSI6CiAgICAgICAgICAgICAgICBjb2xzID0gc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKTsgCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scyk8MjogcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSBhdCBsZWFzdCAyIGNvbHVtbnMgZm9yIGNvcnJlbGF0aW9uLiIpCiAgICAgICAgICAgICAgICBSLmFkZF90YWJsZShFLmNvcnJlbGF0aW9ucyhjb2xzLCAna2VuZGFsbCcpLnJvdW5kKDQpLCJLZW5kYWxsIGNvcnJlbGF0aW9uIikKCiAgICAgICAgICAgIGVsaWYgcD09Ik9MUyByZWdyZXNzaW9uIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBmb3IgT0xTLiIpCiAgICAgICAgICAgICAgICBtPUUub2xzKHZhbChzZWxmLmNtYl95KSwgc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKSk7IFIuYWRkX3RhYmxlKG0uc3VtbWFyeTIoKS50YWJsZXNbMV0ucmVzZXRfaW5kZXgoKSwgIk9MUyBjb2VmZmljaWVudHMiKQoKICAgICAgICAgICAgZWxpZiBwPT0iTG9naXN0aWMgcmVncmVzc2lvbiI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgZm9yIGxvZ2lzdGljLiIpCiAgICAgICAgICAgICAgICBtPUUubG9naXQodmFsKHNlbGYuY21iX3kpLCBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpKTsgdGFiID0gbS5zdW1tYXJ5MigpLnRhYmxlc1sxXS5yZXNldF9pbmRleCgpOyB0YWJbIk9SIl09bnAuZXhwKHRhYlsiQ29lZi4iXSk7IFIuYWRkX3RhYmxlKHRhYiwgIkxvZ2lzdGljIGNvZWZmaWNpZW50cyAoT1IpIikKCiAgICAgICAgICAgIGVsaWYgcD09Ik11bHRpbm9taWFsIGxvZ2l0IjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBmb3IgbXVsdGlub21pYWwuIikKICAgICAgICAgICAgICAgIG09RS5tbG9naXQodmFsKHNlbGYuY21iX3kpLCBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpKQogICAgICAgICAgICAgICAgZm9yIGksdCBpbiBlbnVtZXJhdGUobS5zdW1tYXJ5KCkudGFibGVzKToKICAgICAgICAgICAgICAgICAgICBpZiBoYXNhdHRyKHQsJ2RhdGEnKToKICAgICAgICAgICAgICAgICAgICAgICAgaW1wb3J0IHBhbmRhcyBhcyBwZDsgUi5hZGRfdGFibGUocGQuRGF0YUZyYW1lKHQuZGF0YVsxOl0sIGNvbHVtbnM9dC5kYXRhWzBdKSwgZiJNTkxvZ2l0IHRhYmxlIHtpKzF9IikKCiAgICAgICAgICAgIGVsaWYgcD09Ik9yZGluYWwgcmVncmVzc2lvbiAobG9naXQpIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBmb3Igb3JkaW5hbCByZWdyZXNzaW9uLiIpCiAgICAgICAgICAgICAgICBtPUUub3JkaW5hbCh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSkpOyBSLmFkZF90YWJsZShtLnN1bW1hcnkoKS50YWJsZXNbMV0sICJPcmRlcmVkIGxvZ2l0IGNvZWZmaWNpZW50cyIpCgogICAgICAgICAgICBlbGlmIHA9PSJHTE0gKFBvaXNzb24pIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBmb3IgR0xNLiIpCiAgICAgICAgICAgICAgICBtPUUuZ2xtKHZhbChzZWxmLmNtYl95KSwgc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKSwgJ3BvaXNzb24nKTsgUi5hZGRfdGFibGUobS5zdW1tYXJ5MigpLnRhYmxlc1sxXS5yZXNldF9pbmRleCgpLCAiUG9pc3NvbiBHTE0iKQoKICAgICAgICAgICAgZWxpZiBwPT0iR0xNIChOZWdhdGl2ZSBiaW5vbWlhbCkiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KTogcmFpc2UgVmFsdWVFcnJvcigiUGljayBZIGZvciBHTE0uIikKICAgICAgICAgICAgICAgIG09RS5nbG0odmFsKHNlbGYuY21iX3kpLCBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpLCAnbmVnYmluJyk7IFIuYWRkX3RhYmxlKG0uc3VtbWFyeTIoKS50YWJsZXNbMV0ucmVzZXRfaW5kZXgoKSwgIk5lZ2F0aXZlIGJpbm9taWFsIEdMTSIpCgogICAgICAgICAgICBlbGlmIHA9PSJHTE0gKEdhbW1hKSI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgZm9yIEdMTS4iKQogICAgICAgICAgICAgICAgbT1FLmdsbSh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSksICdnYW1tYScpOyBSLmFkZF90YWJsZShtLnN1bW1hcnkyKCkudGFibGVzWzFdLnJlc2V0X2luZGV4KCksICJHYW1tYSBHTE0gKGxvZyBsaW5rKSIpCgogICAgICAgICAgICBlbGlmIHA9PSJMaW5lYXIgbWl4ZWQgbW9kZWwiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KSBvciBub3QgdmFsKHNlbGYuY21iX2NsdXN0ZXIpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgYW5kIENsdXN0ZXIgZm9yIExNTS4iKQogICAgICAgICAgICAgICAgbT1FLmxtbSh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSksIHZhbChzZWxmLmNtYl9jbHVzdGVyKSk7IFIuYWRkX3RhYmxlKG0uc3VtbWFyeSgpLnRhYmxlc1sxXSwgIkxNTSBmaXhlZCBlZmZlY3RzIikKCiAgICAgICAgICAgIGVsaWYgcD09IkNyb25iYWNoIGFscGhhIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scyk8MjogcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSBhdCBsZWFzdCAyIGl0ZW0gY29sdW1ucyBmb3IgYWxwaGEuIikKICAgICAgICAgICAgICAgIFIuYWRkX2t2KCJDcm9uYmFjaCBhbHBoYSIsIHsiYWxwaGEiOiByb3VuZChmbG9hdChFLmNyb25iYWNoX2FscGhhKGNvbHMpKSw0KX0pCgogICAgICAgICAgICBlbGlmIHA9PSJDb2hlbiBrYXBwYSI6CiAgICAgICAgICAgICAgICB5ID0gdmFsKHNlbGYuY21iX3kpOyB4ID0gdmFsKHNlbGYuY21iX3gpCiAgICAgICAgICAgICAgICBpZiBub3QgeSBvciBub3QgeDogcmFpc2UgVmFsdWVFcnJvcigiU2VsZWN0IGJvdGggWSBhbmQgWCBjb2x1bW5zIGZvciBDb2hlbidzIGthcHBhLiIpCiAgICAgICAgICAgICAgICBSLmFkZF9rdigiQ29oZW4ga2FwcGEiLCB7ImthcHBhIjogcm91bmQoZmxvYXQoRS5rYXBwYSh5LCB4KSksNCl9KQoKICAgICAgICAgICAgZWxpZiBwPT0iV2VpZ2h0ZWQga2FwcGEgKGxpbmVhcikiOgogICAgICAgICAgICAgICAgeSA9IHZhbChzZWxmLmNtYl95KTsgeCA9IHZhbChzZWxmLmNtYl94KQogICAgICAgICAgICAgICAgaWYgbm90IHkgb3Igbm90IHg6IHJhaXNlIFZhbHVlRXJyb3IoIlNlbGVjdCBib3RoIFkgYW5kIFggY29sdW1ucyBmb3Igd2VpZ2h0ZWQga2FwcGEuIikKICAgICAgICAgICAgICAgIFIuYWRkX2t2KCJXZWlnaHRlZCBDb2hlbiBrYXBwYSAobGluZWFyKSIsIHsia2FwcGFfd19saW5lYXIiOiByb3VuZChmbG9hdChFLndlaWdodGVkX2thcHBhKHksIHgsIHdlaWdodHM9J2xpbmVhcicpKSw0KX0pCgogICAgICAgICAgICBlbGlmIHA9PSJXZWlnaHRlZCBrYXBwYSAocXVhZHJhdGljKSI6CiAgICAgICAgICAgICAgICB5ID0gdmFsKHNlbGYuY21iX3kpOyB4ID0gdmFsKHNlbGYuY21iX3gpCiAgICAgICAgICAgICAgICBpZiBub3QgeSBvciBub3QgeDogcmFpc2UgVmFsdWVFcnJvcigiU2VsZWN0IGJvdGggWSBhbmQgWCBjb2x1bW5zIGZvciB3ZWlnaHRlZCBrYXBwYS4iKQogICAgICAgICAgICAgICAgUi5hZGRfa3YoIldlaWdodGVkIENvaGVuIGthcHBhIChxdWFkcmF0aWMpIiwgeyJrYXBwYV93X3F1YWRyYXRpYyI6IHJvdW5kKGZsb2F0KEUud2VpZ2h0ZWRfa2FwcGEoeSwgeCwgd2VpZ2h0cz0ncXVhZHJhdGljJykpLDQpfSkKCiAgICAgICAgICAgIGVsaWYgcD09IkZsZWlzcycga2FwcGEgKG11bHRpLXJhdGVyKSI6CiAgICAgICAgICAgICAgICBjb2xzID0gc2VsZi5fcGFyc2VfY29scyhzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSkKICAgICAgICAgICAgICAgIGlmIGxlbihjb2xzKSA8IDI6CiAgICAgICAgICAgICAgICAgICAgeSA9IHZhbChzZWxmLmNtYl95KTsgeCA9IHZhbChzZWxmLmNtYl94KQogICAgICAgICAgICAgICAgICAgIGlmIHkgYW5kIHg6CiAgICAgICAgICAgICAgICAgICAgICAgIGsgPSBFLnNjb3R0X3BpKHksIHgpCiAgICAgICAgICAgICAgICAgICAgICAgIFIuYWRkX2t2KCJTY290dCdzIHBpIChhdXRvOiB1c2VkIFkgYW5kIFgpIiwgeyJwaSI6IHJvdW5kKGZsb2F0KGspLCA0KSwgInJhdGVyMSI6IHksICJyYXRlcjIiOiB4fSkKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAgICBSLmFkZF9pbmZvKCJGbGVpc3MnIGthcHBhIiwgIlByb3ZpZGUgYXQgbGVhc3QgMiByYXRlciBjb2x1bW5zIGluIFByZWRpY3RvcnMgT1Igc2V0IFkgYW5kIFggKHR3byByYXRlcnMpLiBXaXRoIGV4YWN0bHkgMiwgcmVzdWx0ID0gU2NvdHQncyDPgC4iKQogICAgICAgICAgICAgICAgZWxpZiBsZW4oY29scykgPT0gMjoKICAgICAgICAgICAgICAgICAgICBrID0gRS5zY290dF9waShjb2xzWzBdLCBjb2xzWzFdKQogICAgICAgICAgICAgICAgICAgIFIuYWRkX2t2KCJTY290dCdzIHBpIChGbGVpc3Mgd2l0aCAyIHJhdGVycykiLCB7InBpIjogcm91bmQoZmxvYXQoayksIDQpLCAicmF0ZXIxIjogY29sc1swXSwgInJhdGVyMiI6IGNvbHNbMV19KQogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICBrYXBwYSwgbl9zdWJqLCBuX3JhdGVycywgbl9jYXRzID0gRS5mbGVpc3Nfa2FwcGEoY29scykKICAgICAgICAgICAgICAgICAgICBSLmFkZF9rdigiRmxlaXNzJyBrYXBwYSAobXVsdGktcmF0ZXIpIiwgewogICAgICAgICAgICAgICAgICAgICAgICAia2FwcGEiOiByb3VuZChrYXBwYSwgNCksCiAgICAgICAgICAgICAgICAgICAgICAgICJzdWJqZWN0cyAocm93cyB1c2VkKSI6IG5fc3ViaiwKICAgICAgICAgICAgICAgICAgICAgICAgInJhdGVycyAoY29sdW1ucykiOiBuX3JhdGVycywKICAgICAgICAgICAgICAgICAgICAgICAgImNhdGVnb3JpZXMgZGV0ZWN0ZWQiOiBuX2NhdHMKICAgICAgICAgICAgICAgICAgICB9KQoKICAgICAgICAgICAgZWxpZiBwPT0iSUNDIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9wYXJzZV9jb2xzKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKQogICAgICAgICAgICAgICAgaWYgbGVuKGNvbHMpIDwgMjoKICAgICAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJGb3IgSUNDLCBsaXN0IDIrIHJhdGVyIGNvbHVtbnMgaW4gUHJlZGljdG9ycyAoZS5nLiwgcjEscjIscjMpLiIpCiAgICAgICAgICAgICAgICBSLmFkZF90YWJsZShFLmljYyhjb2xzKS5yb3VuZCg2KSwgIkludHJhY2xhc3MgY29ycmVsYXRpb24iKQoKICAgICAgICAgICAgZWxpZiBwPT0iS3JpcHBlbmRvcmZm4oCZcyBhbHBoYSAobm9taW5hbCkiOgogICAgICAgICAgICAgICAgY29scyA9IHNlbGYuX3BhcnNlX2NvbHMoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scykgPCAyOgogICAgICAgICAgICAgICAgICAgIHkgPSB2YWwoc2VsZi5jbWJfeSk7IHggPSB2YWwoc2VsZi5jbWJfeCkKICAgICAgICAgICAgICAgICAgICBpZiB5IGFuZCB4OgogICAgICAgICAgICAgICAgICAgICAgICBjb2xzID0gW3ksIHhdCiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSAyKyByYXRlciBjb2x1bW5zIGluIFByZWRpY3RvcnMgb3Igc2V0IFkgYW5kIFggKHR3byByYXRlcnMpLiIpCiAgICAgICAgICAgICAgICBhID0gRS5rcmlwcF9hbHBoYShjb2xzLCBsZXZlbD0ibm9taW5hbCIpOyBSLmFkZF9rdigiS3JpcHBlbmRvcmZm4oCZcyDOsSAobm9taW5hbCkiLCB7ImFscGhhIjogcm91bmQoZmxvYXQoYSksNCksICJyYXRlcnMiOiAiLCAiLmpvaW4oY29scyl9KQoKICAgICAgICAgICAgZWxpZiBwPT0iS3JpcHBlbmRvcmZm4oCZcyBhbHBoYSAob3JkaW5hbCkiOgogICAgICAgICAgICAgICAgY29scyA9IHNlbGYuX3BhcnNlX2NvbHMoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scykgPCAyOgogICAgICAgICAgICAgICAgICAgIHkgPSB2YWwoc2VsZi5jbWJfeSk7IHggPSB2YWwoc2VsZi5jbWJfeCkKICAgICAgICAgICAgICAgICAgICBpZiB5IGFuZCB4OgogICAgICAgICAgICAgICAgICAgICAgICBjb2xzID0gW3ksIHhdCiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSAyKyByYXRlciBjb2x1bW5zIGluIFByZWRpY3RvcnMgb3Igc2V0IFkgYW5kIFggKHR3byByYXRlcnMpLiIpCiAgICAgICAgICAgICAgICBhID0gRS5rcmlwcF9hbHBoYShjb2xzLCBsZXZlbD0ib3JkaW5hbCIpOyBSLmFkZF9rdigiS3JpcHBlbmRvcmZm4oCZcyDOsSAob3JkaW5hbCkiLCB7ImFscGhhIjogcm91bmQoZmxvYXQoYSksNCksICJyYXRlcnMiOiAiLCAiLmpvaW4oY29scyl9KQoKICAgICAgICAgICAgZWxpZiBwPT0iS3JpcHBlbmRvcmZm4oCZcyBhbHBoYSAoaW50ZXJ2YWwpIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9wYXJzZV9jb2xzKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKQogICAgICAgICAgICAgICAgaWYgbGVuKGNvbHMpIDwgMjoKICAgICAgICAgICAgICAgICAgICB5ID0gdmFsKHNlbGYuY21iX3kpOyB4ID0gdmFsKHNlbGYuY21iX3gpCiAgICAgICAgICAgICAgICAgICAgaWYgeSBhbmQgeDoKICAgICAgICAgICAgICAgICAgICAgICAgY29scyA9IFt5LCB4XQogICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlByb3ZpZGUgMisgcmF0ZXIgY29sdW1ucyBpbiBQcmVkaWN0b3JzIG9yIHNldCBZIGFuZCBYICh0d28gcmF0ZXJzKS4iKQogICAgICAgICAgICAgICAgYSA9IEUua3JpcHBfYWxwaGEoY29scywgbGV2ZWw9ImludGVydmFsIik7IFIuYWRkX2t2KCJLcmlwcGVuZG9yZmbigJlzIM6xIChpbnRlcnZhbCkiLCB7ImFscGhhIjogcm91bmQoZmxvYXQoYSksNCksICJyYXRlcnMiOiAiLCAiLmpvaW4oY29scyl9KQoKICAgICAgICAgICAgZWxpZiBwPT0iU3Vydml2YWwgKEtNICsgQ294ICsgV2VpYnVsbCBBRlQpIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfdGltZSkgb3Igbm90IHZhbChzZWxmLmNtYl9ldmVudCk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgVGltZSBhbmQgRXZlbnQgZm9yIHN1cnZpdmFsLiIpCiAgICAgICAgICAgICAgICBFLnN1cnZpdmFsKHZhbChzZWxmLmNtYl90aW1lKSwgdmFsKHNlbGYuY21iX2V2ZW50KSwgdmFsKHNlbGYuY21iX2dyb3VwKSwgc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKSwgUikKCiAgICAgICAgICAgIGVsaWYgcD09IlJPQyAvIFBSIC8gQnJpZXIiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KSBvciBub3QgdmFsKHNlbGYuY21iX3gpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgKGJpbmFyeSB0cnVlKSBhbmQgWCAoc2NvcmUpIGZvciBST0MvUFIuIikKICAgICAgICAgICAgICAgIEUuZGlhZ25vc3RpY19jdXJ2ZXModmFsKHNlbGYuY21iX3kpLCB2YWwoc2VsZi5jbWJfeCksIFIpCgogICAgICAgICAgICBlbGlmIHA9PSJBUklNQSBmb3JlY2FzdCI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3RpbWUpIG9yIG5vdCAodmFsKHNlbGYuY21iX3kpIG9yIHZhbChzZWxmLmNtYl94KSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgVGltZSBhbmQgWS9YIGZvciBBUklNQS4iKQogICAgICAgICAgICAgICAgRS5hcmltYV9mb3JlY2FzdCh2YWwoc2VsZi5jbWJfdGltZSksIHZhbChzZWxmLmNtYl95KSBvciB2YWwoc2VsZi5jbWJfeCksIHNlbGYuc3BuX3N0ZXBzLnZhbHVlKCksIFIpCgogICAgICAgICAgICBlbGlmIHA9PSJFVFMgZm9yZWNhc3QiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl90aW1lKSBvciBub3QgKHZhbChzZWxmLmNtYl95KSBvciB2YWwoc2VsZi5jbWJfeCkpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFRpbWUgYW5kIFkvWCBmb3IgRVRTLiIpCiAgICAgICAgICAgICAgICBFLmV0c19mb3JlY2FzdCh2YWwoc2VsZi5jbWJfdGltZSksIHZhbChzZWxmLmNtYl95KSBvciB2YWwoc2VsZi5jbWJfeCksIHNlbGYuc3BuX3N0ZXBzLnZhbHVlKCksIFIsIHNlYXNvbmFsPU5vbmUpCgogICAgICAgICAgICBlbGlmIHA9PSJNZXRhLWFuYWx5c2lzIChETCByYW5kb20tZWZmZWN0cykiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KSBvciBub3QgdmFsKHNlbGYuY21iX3gpOiByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIHlpIChlZmZlY3QpIGluIFkgYW5kIHZpICh2YXJpYW5jZSkgaW4gWC4iKQogICAgICAgICAgICAgICAgRS5tZXRhX2RsKHZhbChzZWxmLmNtYl95KSwgdmFsKHNlbGYuY21iX3gpLCB2YWwoc2VsZi5jbWJfZ3JvdXApLCBSKQoKICAgICAgICAgICAgZWxpZiBwPT0iSVBUVyAoQVRFKSI6CiAgICAgICAgICAgICAgICB5PXZhbChzZWxmLmNtYl95KTsgYT12YWwoc2VsZi5jbWJfZ3JvdXApOyBjb3Y9c2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKQogICAgICAgICAgICAgICAgaWYgbm90IHkgb3Igbm90IGEgb3Igbm90IGNvdjogcmFpc2UgVmFsdWVFcnJvcigiWSwgdHJlYXRtZW50IEdyb3VwLCBhbmQgY292YXJpYXRlcyBhcmUgcmVxdWlyZWQgZm9yIElQVFcuIikKICAgICAgICAgICAgICAgIGQgPSBzZWxmLmRmW1t5LCBhXStjb3ZdLmRyb3BuYSgpLmNvcHkoKTsgWD1kW2Nvdl0udmFsdWVzOyBUPWRbYV0uYXN0eXBlKGludCkudmFsdWVzOyBZPWRbeV0uYXN0eXBlKGludCkudmFsdWVzCiAgICAgICAgICAgICAgICBsciA9IExvZ2lzdGljUmVncmVzc2lvbihtYXhfaXRlcj0yMDApLmZpdChYLFQpOyBwcyA9IG5wLmNsaXAobHIucHJlZGljdF9wcm9iYShYKVs6LDFdLCAxZS0zLCAxLTFlLTMpCiAgICAgICAgICAgICAgICB3ID0gVC9wcyArICgxLVQpLygxLXBzKTsgbWRsID0gc20uR0xNKFksIHNtLmFkZF9jb25zdGFudChUKSwgZmFtaWx5PXNtLmZhbWlsaWVzLkJpbm9taWFsKCksIGZyZXFfd2VpZ2h0cz13KS5maXQoKQogICAgICAgICAgICAgICAgT1I9bnAuZXhwKG1kbC5wYXJhbXNbMV0pOyBsbyxoaT1ucC5leHAobWRsLmNvbmZfaW50KCkubG9jWzFdKTsgUi5hZGRfa3YoIklQVFcgKEFURSkgd2VpZ2h0ZWQgbG9naXQiLCB7Ik9SIjogcm91bmQoZmxvYXQoT1IpLDQpLCAiQ0kgbG93Ijogcm91bmQoZmxvYXQobG8pLDQpLCAiQ0kgaGlnaCI6IHJvdW5kKGZsb2F0KGhpKSw0KX0pCgogICAgICAgICAgICBlbGlmIHA9PSJEaWZmZXJlbmNlLWluLURpZmZlcmVuY2VzIjoKICAgICAgICAgICAgICAgIHk9dmFsKHNlbGYuY21iX3kpOyB0cj12YWwoc2VsZi5jbWJfZ3JvdXApOyB0bT12YWwoc2VsZi5jbWJfZ3JvdXAyKTsgY292PXNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSkKICAgICAgICAgICAgICAgIGlmIG5vdCB5IG9yIG5vdCB0ciBvciBub3QgdG06IHJhaXNlIFZhbHVlRXJyb3IoIlksIHRyZWF0bWVudCwgYW5kIHRpbWUgZ3JvdXAgY29sdW1ucyBhcmUgcmVxdWlyZWQgZm9yIERpRC4iKQogICAgICAgICAgICAgICAgZm9ybXVsYT1mInt5fSB+IHt0cn0gKyB7dG19ICsge3RyfTp7dG19IiArICgoIiArICIgKyAiICsgIi5qb2luKGNvdikpIGlmIGNvdiBlbHNlICIiKQogICAgICAgICAgICAgICAgbT1zbWYub2xzKGZvcm11bGEsIGRhdGE9c2VsZi5kZi5kcm9wbmEoc3Vic2V0PVt5LHRyLHRtXStjb3YpKS5maXQoKTsgUi5hZGRfdGFibGUobS5zdW1tYXJ5MigpLnRhYmxlc1sxXS5yZXNldF9pbmRleCgpLCAiRGlEIGNvZWZmaWNpZW50cyIpCgogICAgICAgICAgICBlbGlmIHA9PSJQQ0EiOgogICAgICAgICAgICAgICAgY29scyA9IHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSk7IAogICAgICAgICAgICAgICAgaWYgbGVuKGNvbHMpPDI6IHJhaXNlIFZhbHVlRXJyb3IoIlByb3ZpZGUgYXQgbGVhc3QgMiBudW1lcmljIGNvbHVtbnMgZm9yIFBDQS4iKQogICAgICAgICAgICAgICAgRS5wY2EoY29scywgc2VsZi5zcG5fay52YWx1ZSgpLCBSKQoKICAgICAgICAgICAgZWxpZiBwPT0iRUZBICh2YXJpbWF4KSI6CiAgICAgICAgICAgICAgICBjb2xzID0gc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKTsgCiAgICAgICAgICAgICAgICBpZiBsZW4oY29scyk8MjogcmFpc2UgVmFsdWVFcnJvcigiUHJvdmlkZSBhdCBsZWFzdCAyIG51bWVyaWMgY29sdW1ucyBmb3IgRUZBLiIpCiAgICAgICAgICAgICAgICBFLmVmYShjb2xzLCBzZWxmLnNwbl9rLnZhbHVlKCksIFIpCgogICAgICAgICAgICBlbGlmIHA9PSJDb3JyZXNwb25kZW5jZSBhbmFseXNpcyI6CiAgICAgICAgICAgICAgICBhID0gdmFsKHNlbGYuY21iX2dyb3VwKTsgYiA9IHZhbChzZWxmLmNtYl9ncm91cDIpCiAgICAgICAgICAgICAgICBpZiBub3QgYSBvciBub3QgYjogcmFpc2UgVmFsdWVFcnJvcigiUGljayB0d28gY2F0ZWdvcmljYWwgY29sdW1ucyBmb3IgY29ycmVzcG9uZGVuY2UgYW5hbHlzaXMuIikKICAgICAgICAgICAgICAgIEUuY29ycmVzcG9uZGVuY2UoYSwgYiwgUikKCiAgICAgICAgICAgIGVsaWYgcD09IkxEQSAoY2xhc3NpZmljYXRpb24pIjoKICAgICAgICAgICAgICAgIGlmIG5vdCB2YWwoc2VsZi5jbWJfeSk6IHJhaXNlIFZhbHVlRXJyb3IoIlBpY2sgWSBmb3IgTERBLiIpCiAgICAgICAgICAgICAgICBFLmxkYSh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSksIFIpCgogICAgICAgICAgICBlbGlmIHA9PSJEZWNpc2lvbiB0cmVlIChjbGFzc2lmaWNhdGlvbikiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KTogcmFpc2UgVmFsdWVFcnJvcigiUGljayBZIGZvciBkZWNpc2lvbiB0cmVlLiIpCiAgICAgICAgICAgICAgICBFLnRyZWVfY2xzKHZhbChzZWxmLmNtYl95KSwgc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKSwgUiwgY3JpdGVyaW9uPSJnaW5pIikKCiAgICAgICAgICAgIGVsaWYgcD09IkNIQUlELWxpa2UgdHJlZSI6CiAgICAgICAgICAgICAgICBpZiBub3QgdmFsKHNlbGYuY21iX3kpOiByYWlzZSBWYWx1ZUVycm9yKCJQaWNrIFkgZm9yIENIQUlELWxpa2UgdHJlZS4iKQogICAgICAgICAgICAgICAgRS50cmVlX2Nscyh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSksIFIsIGNyaXRlcmlvbj0iZW50cm9weSIpCgogICAgICAgICAgICBlbGlmIHA9PSJSYW5kb20gZm9yZXN0IChjbGFzc2lmaWNhdGlvbikiOgogICAgICAgICAgICAgICAgaWYgbm90IHZhbChzZWxmLmNtYl95KTogcmFpc2UgVmFsdWVFcnJvcigiUGljayBZIGZvciByYW5kb20gZm9yZXN0LiIpCiAgICAgICAgICAgICAgICBFLnJmX2Nscyh2YWwoc2VsZi5jbWJfeSksIHNlbGYuX3NwbGl0X3NpbXBsZShzZWxmLnR4dF9wcmVkaWN0b3JzLnRleHQoKSksIFIpCgogICAgICAgICAgICBlbGlmIHA9PSJLLU1lYW5zIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpOyAKICAgICAgICAgICAgICAgIGlmIGxlbihjb2xzKTwyOiByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIG51bWVyaWMgY29sdW1ucyBmb3IgSy1NZWFucy4iKQogICAgICAgICAgICAgICAgRS5rbWVhbnMoY29scywgc2VsZi5zcG5fay52YWx1ZSgpLCBSKQoKICAgICAgICAgICAgZWxpZiBwPT0iQWdnbG9tZXJhdGl2ZSBjbHVzdGVyaW5nIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpOyAKICAgICAgICAgICAgICAgIGlmIGxlbihjb2xzKTwyOiByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIG51bWVyaWMgY29sdW1ucyBmb3IgQWdnbG9tZXJhdGl2ZSBjbHVzdGVyaW5nLiIpCiAgICAgICAgICAgICAgICBFLmFnZ2xvbWVyYXRpdmUoY29scywgc2VsZi5zcG5fay52YWx1ZSgpLCBSKQoKICAgICAgICAgICAgZWxpZiBwPT0iQXV0by1LTWVhbnMgKFR3b1N0ZXAtaW5zcGlyZWQpIjoKICAgICAgICAgICAgICAgIGNvbHMgPSBzZWxmLl9zcGxpdF9zaW1wbGUoc2VsZi50eHRfcHJlZGljdG9ycy50ZXh0KCkpOyAKICAgICAgICAgICAgICAgIGlmIGxlbihjb2xzKTwyOiByYWlzZSBWYWx1ZUVycm9yKCJQcm92aWRlIG51bWVyaWMgY29sdW1ucyBmb3IgQXV0by1LTWVhbnMuIikKICAgICAgICAgICAgICAgIEUuYXV0b19rbWVhbnMoY29scywgUikKCiAgICAgICAgICAgIGVsaWYgcD09Ik1BTk9WQSAvIE1BTkNPVkEiOgogICAgICAgICAgICAgICAgeXMgPSBbdC5zdHJpcCgpIGZvciB0IGluIChzZWxmLnR4dF9tdWx0aV95LnRleHQoKSBvciAiIikuc3BsaXQoIiwiKSBpZiB0LnN0cmlwKCldCiAgICAgICAgICAgICAgICByZXMgPSBFLm1hbm92YSh5cywgc2VsZi5fc3BsaXRfc2ltcGxlKHNlbGYudHh0X3ByZWRpY3RvcnMudGV4dCgpKSk7IFIuYWRkX2luZm8oIk1BTk9WQSAvIE1BTkNPVkEiLCBmIjxwcmU+e3Jlc308L3ByZT4iKQoKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlVua25vd24gcHJvY2VkdXJlLiIpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBSLmFkZF9pbmZvKCJSdW4gZXJyb3IiLCBmIntlfTxicj48cHJlPnt0cmFjZWJhY2suZm9ybWF0X2V4YygpfTwvcHJlPiIpCgogICAgICAgIHNlbGYucmVwb3J0ID0gUjsgc2VsZi5fcmVmcmVzaCgpCgogICAgCiAgICBkZWYgX3JlZnJlc2goc2VsZik6CiAgICAgICAgaHRtbCA9IHNlbGYucmVwb3J0Lmh0bWwoKTsgCiAgICAgICAgc2VsZi50eHRfcmVwb3J0LnNldEh0bWwoaHRtbCkKICAgICAgICBpZiBoYXNhdHRyKHNlbGYsICJ0eHRfZmluYWwiKToKICAgICAgICAgICAgc2VsZi50eHRfZmluYWwuc2V0SHRtbChodG1sKQoKICAgIGRlZiBleHBvcnRfaHRtbChzZWxmKToKICAgICAgICBwLF8gPSBRRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwiU2F2ZSBIVE1MIiwgb3MucGF0aC5qb2luKG9zLmdldGN3ZCgpLCJyZXBvcnQuaHRtbCIpLCAiSFRNTCAoKi5odG1sKSIpCiAgICAgICAgaWYgbm90IHA6IHJldHVybgogICAgICAgIHdpdGggb3BlbihwLCd3JyxlbmNvZGluZz0ndXRmLTgnKSBhcyBmOiBmLndyaXRlKHNlbGYucmVwb3J0Lmh0bWwoKSkKICAgICAgICBRTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCJTYXZlZCIsZiJTYXZlZCB0byB7cH0iKQoKICAgIGRlZiBleHBvcnRfZG9jeChzZWxmKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIGZyb20gZG9jeCBpbXBvcnQgRG9jdW1lbnQKICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICBRTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCJET0NYIHVuYXZhaWxhYmxlIiwiSW5zdGFsbCBweXRob24tZG9jeCB0byBleHBvcnQgRE9DWC4iKTsgcmV0dXJuCiAgICAgICAgcCxfID0gUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsIlNhdmUgRE9DWCIsIG9zLnBhdGguam9pbihvcy5nZXRjd2QoKSwicmVwb3J0LmRvY3giKSwgIkRPQ1ggKCouZG9jeCkiKQogICAgICAgIGlmIG5vdCBwOiByZXR1cm4KICAgICAgICBkb2MgPSBEb2N1bWVudCgpOyBkb2MuYWRkX2hlYWRpbmcoQVBQX05BTUUsIGxldmVsPTEpOyBkb2MuYWRkX3BhcmFncmFwaChDT1BZUklHSFQpCiAgICAgICAgZm9yIHBhcmEgaW4gc2VsZi50eHRfcmVwb3J0LnRvUGxhaW5UZXh0KCkuc3BsaXQoIlxuXG4iKTogZG9jLmFkZF9wYXJhZ3JhcGgocGFyYSkKICAgICAgICBkb2Muc2F2ZShwKTsgUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwiU2F2ZWQiLGYiU2F2ZWQgdG8ge3B9IikKCgpkZWYgbWFpbigpOgogICAgYXBwID0gUUFwcGxpY2F0aW9uKHN5cy5hcmd2KTsgYXBwLnNldEFwcGxpY2F0aW9uTmFtZShBUFBfTkFNRSk7IGFwcC5zZXRTdHlsZSgiRnVzaW9uIikKICAgIHBhbCA9IGFwcC5wYWxldHRlKCk7IHBhbC5zZXRDb2xvcihRdEd1aS5RUGFsZXR0ZS5XaW5kb3csIFF0R3VpLlFDb2xvcigyNDgsMjUwLDI1MikpOyBwYWwuc2V0Q29sb3IoUXRHdWkuUVBhbGV0dGUuQmFzZSwgUXQud2hpdGUpOyBhcHAuc2V0UGFsZXR0ZShwYWwpCiAgICB3ID0gTWFpbldpbmRvdygpOyB3LnNob3coKTsgc3lzLmV4aXQoYXBwLmV4ZWMoKSkKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK'
_EMTAS_B64='aW1wb3J0IG9zLCBqc29uLCBtYXRoLCBjc3YsIGRhdGV0aW1lCmltcG9ydCBudW1weSBhcyBucAp0cnk6CiAgICBmcm9tIFB5UXQ1IGltcG9ydCBRdFdpZGdldHMsIFF0Q29yZSwgUXRHdWkKICAgIGZyb20gbWF0cGxvdGxpYi5iYWNrZW5kcy5iYWNrZW5kX3F0NWFnZyBpbXBvcnQgRmlndXJlQ2FudmFzUVRBZ2cgYXMgRmlndXJlQ2FudmFzCmV4Y2VwdCBJbXBvcnRFcnJvcjoKICAgIHRyeToKICAgICAgICBmcm9tIFB5U2lkZTYgaW1wb3J0IFF0V2lkZ2V0cywgUXRDb3JlLCBRdEd1aQogICAgICAgIGZyb20gbWF0cGxvdGxpYi5iYWNrZW5kcy5iYWNrZW5kX3F0YWdnIGltcG9ydCBGaWd1cmVDYW52YXNRVEFnZyBhcyBGaWd1cmVDYW52YXMKICAgIGV4Y2VwdCBJbXBvcnRFcnJvciBhcyBlOgogICAgICAgIHByaW50KCJcbltFUlJPUl0gUXQgYmluZGluZ3Mgbm90IGZvdW5kLiBJbnN0YWxsIG9uZSBvZiB0aGVzZToiKQogICAgICAgIHByaW50KCIgIHBpcCBpbnN0YWxsIFB5UXQ1IG1hdHBsb3RsaWIgbnVtcHkiKQogICAgICAgIHByaW50KCIgIChvcikgcGlwIGluc3RhbGwgUHlTaWRlNiBtYXRwbG90bGliIG51bXB5IikKICAgICAgICByYWlzZQoKaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdAoKCmZyb20gbWV0cmljcyBpbXBvcnQgKAogICAgYmluYXJ5X21ldHJpY3MsIHJvY19wb2ludHNfYXVjLCBwcl9wb2ludHNfYXAsCiAgICBmbGVpc3Nfa2FwcGFfZnJvbV9yYXcsIGJvb3RzdHJhcF9mbGVpc3NfY2ksCiAgICBpY2MyXzEsIGJvb3RzdHJhcF9pY2NfY2ksIGJvb3RzdHJhcF9pY2NfZGlzdHJpYnV0aW9uCikKCkFQUF9USVRMRSA9ICJFTVRBUyB2MS4wIC0gwqkgMjAyNSBNaXJ6YSBOaWF6IFphbWFuIEVsaW4uIEFsbCByaWdodHMgcmVzZXJ2ZWQuIgpFWFBPUlRfRElSID0gImV4cG9ydHMiCgoKZGVmIGVuc3VyZV9leHBvcnRfZGlyKCk6CiAgICBvcy5tYWtlZGlycyhFWFBPUlRfRElSLCBleGlzdF9vaz1UcnVlKQoKCgoKZGVmIHNwZWFybWFuX2Jyb3duKGljYzEsIGspOgogICAgYXJyID0gbnAuYXNhcnJheShpY2MxLCBkdHlwZT1mbG9hdCkKICAgIGlmIGsgPD0gMToKICAgICAgICByZXR1cm4gYXJyCiAgICB3aXRoIG5wLmVycnN0YXRlKGludmFsaWQ9J2lnbm9yZScsIGRpdmlkZT0naWdub3JlJyk6CiAgICAgICAgb3V0ID0gKGsgKiBhcnIpIC8gKDEuMCArIChrIC0gMS4wKSAqIGFycikKICAgIHJldHVybiBvdXQKCgpkZWYgaWNjX2F2ZyhpY2MxLCBrLCByZWR1Y2U9J21lYW4nKToKCiAgIGRlZiBjb25mdXNpb25fY291bnRzKHlfdHJ1ZSwgeV9wcmVkKToKICAgICIiIlJldHVybiBUUCwgRlAsIFROLCBGTiBmb3IgYmluYXJ5IDAvMSBsYWJlbHMuIiIiCiAgICB0cCA9IGZwID0gdG4gPSBmbiA9IDAKICAgIGZvciB5dCwgeXAgaW4gemlwKHlfdHJ1ZSwgeV9wcmVkKToKICAgICAgICBpZiB5dCA9PSAxIGFuZCB5cCA9PSAxOiB0cCArPSAxCiAgICAgICAgZWxpZiB5dCA9PSAwIGFuZCB5cCA9PSAxOiBmcCArPSAxCiAgICAgICAgZWxpZiB5dCA9PSAwIGFuZCB5cCA9PSAwOiB0biArPSAxCiAgICAgICAgZWxpZiB5dCA9PSAxIGFuZCB5cCA9PSAwOiBmbiArPSAxCiAgICByZXR1cm4gdHAsIGZwLCB0biwgZm4KICAgIHNiID0gc3BlYXJtYW5fYnJvd24oaWNjMSwgaykKICAgIGlmIG5wLmlzc2NhbGFyKHNiKSBvciAoaXNpbnN0YW5jZShzYiwgbnAubmRhcnJheSkgYW5kIHNiLm5kaW0gPT0gMCk6CiAgICAgICAgcmV0dXJuIGZsb2F0KHNiKQogICAgaWYgcmVkdWNlID09ICdtZWFuJzoKICAgICAgICByZXR1cm4gZmxvYXQobnAubmFubWVhbihzYikpCiAgICBlbGlmIHJlZHVjZSA9PSAnbWVkaWFuJzoKICAgICAgICByZXR1cm4gZmxvYXQobnAubmFubWVkaWFuKHNiKSkKICAgIGVsc2U6CiAgICAgICAgcmV0dXJuIHNiCgoKCgpkZWYgX2FzX2NsZWFuX2xhYmVscyhzZXEpOgogICAgIiIiQ29udmVydCB2YWx1ZXMgdG8gc3RyaW5ncywga2VlcCBOb25lIGZvciBlbXB0aWVzLiIiIgogICAgb3V0ID0gW10KICAgIGZvciB4IGluIHNlcToKICAgICAgICBpZiB4IGlzIE5vbmU6CiAgICAgICAgICAgIG91dC5hcHBlbmQoTm9uZSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBzID0gc3RyKHgpLnN0cmlwKCkKICAgICAgICAgICAgb3V0LmFwcGVuZChOb25lIGlmIHMgPT0gJycgZWxzZSBzKQogICAgcmV0dXJuIG91dAoKCmRlZiBjb2hlbnNfa2FwcGEobGFiZWxzMSwgbGFiZWxzMik6CiAgICAiIiJDb2hlbidzIGthcHBhIGZvciB0d28gcmF0ZXJzLCBub21pbmFsIGNhdGVnb3JpZXMuCiAgICBsYWJlbHMxL2xhYmVsczI6IHNlcXVlbmNlcyBvZiBzdHJpbmdzL2ludHM7IE5vbmUvJycgaWdub3JlZC4KICAgICIiIgogICAgYSA9IF9hc19jbGVhbl9sYWJlbHMobGFiZWxzMSkKICAgIGIgPSBfYXNfY2xlYW5fbGFiZWxzKGxhYmVsczIpCiAgICBwYWlycyA9IFsoeCwgeSkgZm9yIHgsIHkgaW4gemlwKGEsIGIpIGlmICh4IGlzIG5vdCBOb25lIGFuZCB5IGlzIG5vdCBOb25lKV0KICAgIGlmIG5vdCBwYWlyczoKICAgICAgICByZXR1cm4gZmxvYXQoJ25hbicpLCB7CiAgICAgICAgICAgICdwbyc6IGZsb2F0KCduYW4nKSwgJ3BlJzogZmxvYXQoJ25hbicpLCAnbic6IDAsCiAgICAgICAgICAgICdjYXRlZ29yaWVzJzoge30sICdjb25mdXNpb24nOiB7fQogICAgICAgIH0KICAgIGNhdHMgPSBzb3J0ZWQoe3ggZm9yIHgsIF8gaW4gcGFpcnN9IHwge3kgZm9yIF8sIHkgaW4gcGFpcnN9KQogICAgaWR4ID0ge2M6IGkgZm9yIGksIGMgaW4gZW51bWVyYXRlKGNhdHMpfQogICAgbSA9IG5wLnplcm9zKChsZW4oY2F0cyksIGxlbihjYXRzKSksIGR0eXBlPWZsb2F0KQogICAgZm9yIHgsIHkgaW4gcGFpcnM6CiAgICAgICAgbVtpZHhbeF0sIGlkeFt5XV0gKz0gMQogICAgbiA9IG0uc3VtKCkKICAgIHBvID0gbnAudHJhY2UobSkgLyBuIGlmIG4gZWxzZSBmbG9hdCgnbmFuJykKICAgIHJvdyA9IG0uc3VtKGF4aXM9MSkgLyBuCiAgICBjb2wgPSBtLnN1bShheGlzPTApIC8gbgogICAgcGUgPSBmbG9hdCgocm93IEAgY29sKSkKICAgIGRlbm9tID0gMSAtIHBlCiAgICBrYXBwYSA9IChwbyAtIHBlKSAvIGRlbm9tIGlmIGRlbm9tICE9IDAgZWxzZSBmbG9hdCgnbmFuJykKICAgIAogICAgY29uZiA9IHtjYXRzW2ldOiB7Y2F0c1tqXTogaW50KG1baSwgal0pIGZvciBqIGluIHJhbmdlKGxlbihjYXRzKSl9IGZvciBpIGluIHJhbmdlKGxlbihjYXRzKSl9CiAgICBjYXRfZnJlcSA9IHtjOiBpbnQobVtpLCA6XS5zdW0oKSkgZm9yIGMsIGkgaW4gaWR4Lml0ZW1zKCl9CiAgICByZXR1cm4gZmxvYXQoa2FwcGEpLCB7J3BvJzogZmxvYXQocG8pLCAncGUnOiBmbG9hdChwZSksICduJzogaW50KG4pLCAnY2F0ZWdvcmllcyc6IGNhdF9mcmVxLCAnY29uZnVzaW9uJzogY29uZn0KCgpkZWYga3JpcHBlbmRvcmZmX2FscGhhX25vbWluYWwodGFibGUpOgogICAgIiIiS3JpcHBlbmRvcmZmJ3MgYWxwaGEgKG5vbWluYWwpIGZvciAyKyByYXRlcnMgd2l0aCBtaXNzaW5nIHZhbHVlcy4KICAgIGB0YWJsZWAgaXMgbGlzdC1vZi1saXN0cyBwZXIgdW5pdDogW3JhdGluZ19ieV9yYXRlcjEsIHJhdGluZ19ieV9yYXRlcjIsIC4uLl0uCiAgICBNaXNzaW5nIHZhbHVlczogTm9uZSBvciAnJy4gQWxwaGEgaXMgMSAtIERvL0RlIHVzaW5nIHBhaXIgZGlzYWdyZWVtZW50LgogICAgIiIiCiAgICAKICAgIHVuaXRzID0gW10KICAgIGZvciByb3cgaW4gdGFibGU6CiAgICAgICAgY2xlYW5lZCA9IFtOb25lIGlmICh2IGlzIE5vbmUgb3Igc3RyKHYpLnN0cmlwKCkgPT0gJycpIGVsc2Ugc3RyKHYpLnN0cmlwKCkgZm9yIHYgaW4gcm93XQogICAgICAgIAogICAgICAgIGlmIHN1bSgxIGZvciB2IGluIGNsZWFuZWQgaWYgdiBpcyBub3QgTm9uZSkgPj0gMjoKICAgICAgICAgICAgdW5pdHMuYXBwZW5kKGNsZWFuZWQpCiAgICBpZiBub3QgdW5pdHM6CiAgICAgICAgcmV0dXJuIGZsb2F0KCduYW4nKSwgeydEbyc6IGZsb2F0KCduYW4nKSwgJ0RlJzogZmxvYXQoJ25hbicpLCAndG90YWxfcGFpcnMnOiAwLCAnY2F0ZWdvcnlfZnJlcXVlbmN5Jzoge319CgogICAgCiAgICBjYXRfY291bnRzID0ge30KICAgIHRvdGFsX3JhdGluZ3MgPSAwCiAgICB0b3RhbF9wYWlycyA9IDAKICAgIGRpc2FncmVlX3BhaXJzID0gMAoKICAgIGZvciByb3cgaW4gdW5pdHM6CiAgICAgICAgdmFscyA9IFt2IGZvciB2IGluIHJvdyBpZiB2IGlzIG5vdCBOb25lXQogICAgICAgIHRvdGFsX3JhdGluZ3MgKz0gbGVuKHZhbHMpCiAgICAgICAgZm9yIHYgaW4gdmFsczoKICAgICAgICAgICAgY2F0X2NvdW50c1t2XSA9IGNhdF9jb3VudHMuZ2V0KHYsIDApICsgMQogICAgICAgIAogICAgICAgIG5faSA9IGxlbih2YWxzKQogICAgICAgIHRvdGFsX3BhaXJzICs9IG5faSAqIChuX2kgLSAxKQogICAgICAgIGFncmVlX3BhaXJzX2kgPSAwCiAgICAgICAgCiAgICAgICAgZnJvbSBjb2xsZWN0aW9ucyBpbXBvcnQgQ291bnRlcgogICAgICAgIGN0cyA9IENvdW50ZXIodmFscykKICAgICAgICBmb3IgYywgbl9pYyBpbiBjdHMuaXRlbXMoKToKICAgICAgICAgICAgYWdyZWVfcGFpcnNfaSArPSBuX2ljICogKG5faWMgLSAxKQogICAgICAgIGRpc2FncmVlX3BhaXJzICs9IChuX2kgKiAobl9pIC0gMSkgLSBhZ3JlZV9wYWlyc19pKQoKICAgIGlmIHRvdGFsX3BhaXJzID09IDA6CiAgICAgICAgcmV0dXJuIGZsb2F0KCduYW4nKSwgeydEbyc6IGZsb2F0KCduYW4nKSwgJ0RlJzogZmxvYXQoJ25hbicpLCAndG90YWxfcGFpcnMnOiAwLCAnY2F0ZWdvcnlfZnJlcXVlbmN5JzogY2F0X2NvdW50c30KCiAgICBEbyA9IGRpc2FncmVlX3BhaXJzIC8gdG90YWxfcGFpcnMKICAgIGlmIHRvdGFsX3JhdGluZ3MgPT0gMDoKICAgICAgICByZXR1cm4gZmxvYXQoJ25hbicpLCB7J0RvJzogZmxvYXQoJ25hbicpLCAnRGUnOiBmbG9hdCgnbmFuJyksICd0b3RhbF9wYWlycyc6IHRvdGFsX3BhaXJzLCAnY2F0ZWdvcnlfZnJlcXVlbmN5JzogY2F0X2NvdW50c30KCiAgICAKICAgIHByb2JzID0gW2NudCAvIHRvdGFsX3JhdGluZ3MgZm9yIGNudCBpbiBjYXRfY291bnRzLnZhbHVlcygpXQogICAgRGUgPSAxLjAgLSBmbG9hdChucC5zdW0obnAuc3F1YXJlKHByb2JzKSkpCgogICAgaWYgRGUgPT0gMDoKICAgICAgICBhbHBoYSA9IGZsb2F0KCduYW4nKSAgCiAgICBlbHNlOgogICAgICAgIGFscGhhID0gMS4wIC0gKERvIC8gRGUpCgogICAgcmV0dXJuIGZsb2F0KGFscGhhKSwgeydEbyc6IGZsb2F0KERvKSwgJ0RlJzogZmxvYXQoRGUpLCAndG90YWxfcGFpcnMnOiBpbnQodG90YWxfcGFpcnMpLCAnY2F0ZWdvcnlfZnJlcXVlbmN5JzogY2F0X2NvdW50c30KCgoKCmNsYXNzIExvZ2luRGlhbG9nKFF0V2lkZ2V0cy5RRGlhbG9nKToKICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLnNldFdpbmRvd1RpdGxlKCJTaWduIGluIikKICAgICAgICBzZWxmLnNldE1vZGFsKFRydWUpCiAgICAgICAgbGF5b3V0ID0gUXRXaWRnZXRzLlFGb3JtTGF5b3V0KHNlbGYpCiAgICAgICAgc2VsZi51c2VyID0gUXRXaWRnZXRzLlFMaW5lRWRpdChzZWxmKQogICAgICAgIHNlbGYucGFzc3cgPSBRdFdpZGdldHMuUUxpbmVFZGl0KHNlbGYpOyBzZWxmLnBhc3N3LnNldEVjaG9Nb2RlKFF0V2lkZ2V0cy5RTGluZUVkaXQuUGFzc3dvcmQpCiAgICAgICAgbGF5b3V0LmFkZFJvdygiVXNlcm5hbWUiLCBzZWxmLnVzZXIpCiAgICAgICAgbGF5b3V0LmFkZFJvdygiUGFzc3dvcmQiLCBzZWxmLnBhc3N3KQogICAgICAgIGJ0bnMgPSBRdFdpZGdldHMuUURpYWxvZ0J1dHRvbkJveChRdFdpZGdldHMuUURpYWxvZ0J1dHRvbkJveC5PayB8IFF0V2lkZ2V0cy5RRGlhbG9nQnV0dG9uQm94LkNhbmNlbCkKICAgICAgICBidG5zLmFjY2VwdGVkLmNvbm5lY3Qoc2VsZi5hY2NlcHQpOyBidG5zLnJlamVjdGVkLmNvbm5lY3Qoc2VsZi5yZWplY3QpCiAgICAgICAgbGF5b3V0LmFkZFJvdyhidG5zKQogICAgZGVmIGdldF91c2VyKHNlbGYpOgogICAgICAgIHJldHVybiBzZWxmLnVzZXIudGV4dCgpLnN0cmlwKCkgb3IgIlVzZXIiCgoKCgpjbGFzcyBCaW5hcnlUYWIoUXRXaWRnZXRzLlFXaWRnZXQpOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIHBhcmVudD1Ob25lKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKHBhcmVudCkKCiAgICAgICAgCiAgICAgICAgbmFtZV9yb3cgPSBRdFdpZGdldHMuUUhCb3hMYXlvdXQoKQogICAgICAgIG5hbWVfcm93LmFkZFdpZGdldChRdFdpZGdldHMuUUxhYmVsKCJOYW1lL0dyb3VwOiIpKQogICAgICAgIHNlbGYubmFtZV9lZGl0ID0gUXRXaWRnZXRzLlFMaW5lRWRpdChzZWxmKQogICAgICAgIHNlbGYubmFtZV9lZGl0LnNldFBsYWNlaG9sZGVyVGV4dCgiZS5nLiwgR3JvdXAtMSwgVGVhbSBBLCBSYXRlciBYIikKICAgICAgICBuYW1lX3Jvdy5hZGRXaWRnZXQoc2VsZi5uYW1lX2VkaXQpCgogICAgICAgIHNlbGYudGFibGUgPSBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0KDEwLCA0LCBzZWxmKQogICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhbInN1YmplY3RfaWQiLCJ0cnVlX2xhYmVsIiwicHJlZF9sYWJlbCIsInNjb3JlIl0pCiAgICAgICAgc2VsZi5yZXN1bHQgPSBRdFdpZGdldHMuUVBsYWluVGV4dEVkaXQoc2VsZik7IHNlbGYucmVzdWx0LnNldFJlYWRPbmx5KFRydWUpCgogICAgICAgIAogICAgICAgIHNlbGYuY2FsY19idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkNhbGN1bGF0ZSIpCiAgICAgICAgc2VsZi5ncmFwaF9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkdyYXBoIikKICAgICAgICBzZWxmLnNhdmVfZ3JhcGhfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJTYXZlIEdyYXBoIikKICAgICAgICBzZWxmLnNhdmVfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJTYXZlIERhdGEvSlNPTiIpCiAgICAgICAgc2VsZi5leHBvcnRfcmVzdWx0c19idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkV4cG9ydCBSZXN1bHRzICgudHh0KSIpCiAgICAgICAgc2VsZi5pbXBvcnRfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJJbXBvcnQgQ1NWIikKICAgICAgICBzZWxmLmJhdGNoX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiQmF0Y2ggZnJvbSBGb2xkZXIiKQogICAgICAgIHNlbGYuYWRkX29uZV9yb3dfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJBZGQgMSBpdGVtIikKICAgICAgICBzZWxmLnJlbV9vbmVfcm93X2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiUmVtb3ZlIDEgaXRlbSIpCiAgICAgICAgc2VsZi5hZGRfcm93X2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiQWRkIDEwIHJvd3MiKQogICAgICAgIHNlbGYubG9hZF90cGwgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkxvYWQgdGVtcGxhdGUiKQoKICAgICAgICBidG5fcm93ID0gUXRXaWRnZXRzLlFIQm94TGF5b3V0KCkKICAgICAgICBmb3IgYiBpbiBbc2VsZi5jYWxjX2J0biwgc2VsZi5ncmFwaF9idG4sIHNlbGYuc2F2ZV9ncmFwaF9idG4sIHNlbGYuc2F2ZV9idG4sCiAgICAgICAgICAgICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuLCBzZWxmLmltcG9ydF9idG4sIHNlbGYuYmF0Y2hfYnRuLAogICAgICAgICAgICAgICAgICBzZWxmLmFkZF9vbmVfcm93X2J0biwgc2VsZi5yZW1fb25lX3Jvd19idG4sIHNlbGYuYWRkX3Jvd19idG4sIHNlbGYubG9hZF90cGxdOgogICAgICAgICAgICBidG5fcm93LmFkZFdpZGdldChiKQoKICAgICAgICBsYXlvdXQgPSBRdFdpZGdldHMuUVZCb3hMYXlvdXQoc2VsZikKICAgICAgICBsYXlvdXQuYWRkTGF5b3V0KG5hbWVfcm93KQogICAgICAgIGxheW91dC5hZGRXaWRnZXQoc2VsZi50YWJsZSkKICAgICAgICBsYXlvdXQuYWRkTGF5b3V0KGJ0bl9yb3cpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnJlc3VsdCkKCiAgICAgICAgCiAgICAgICAgc2VsZi5jYWxjX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5jYWxjdWxhdGUpCiAgICAgICAgc2VsZi5ncmFwaF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuZ3JhcGgpCiAgICAgICAgc2VsZi5zYXZlX2dyYXBoX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5zYXZlX2dyYXBoKQogICAgICAgIHNlbGYuc2F2ZV9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuc2F2ZSkKICAgICAgICBzZWxmLmV4cG9ydF9yZXN1bHRzX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5leHBvcnRfcmVzdWx0cykKICAgICAgICBzZWxmLmltcG9ydF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuaW1wb3J0X2NzdikKICAgICAgICBzZWxmLmJhdGNoX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5iYXRjaF9mcm9tX2ZvbGRlcikKICAgICAgICBzZWxmLmFkZF9vbmVfcm93X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5hZGRfb25lX3JvdykKICAgICAgICBzZWxmLnJlbV9vbmVfcm93X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5yZW1fb25lX3JvdykKICAgICAgICBzZWxmLmFkZF9yb3dfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmFkZF9yb3dzKQogICAgICAgIHNlbGYubG9hZF90cGwuY2xpY2tlZC5jb25uZWN0KHNlbGYubG9hZF90ZW1wbGF0ZSkKICAgICAgICBzZWxmLmxhc3RfY2FsYyA9IE5vbmUKCiAgICAKICAgIGRlZiBsb2FkX3RlbXBsYXRlKHNlbGYpOgogICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4oImV4YW1wbGVzIiwiYmluYXJ5X3RlbXBsYXRlLmNzdiIpCiAgICAgICAgaWYgbm90IG9zLnBhdGguZXhpc3RzKHBhdGgpOgogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3gud2FybmluZyhzZWxmLCJUZW1wbGF0ZSIsIlRlbXBsYXRlIG5vdCBmb3VuZC4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLl9sb2FkX2Nzdl90b190YWJsZShwYXRoKQoKICAgIGRlZiBpbXBvcnRfY3N2KHNlbGYpOgogICAgICAgIHBhdGgsIF8gPSBRdFdpZGdldHMuUUZpbGVEaWFsb2cuZ2V0T3BlbkZpbGVOYW1lKHNlbGYsICJJbXBvcnQgQ1NWIiwgIiIsICJDU1YgRmlsZXMgKCouY3N2KSIpCiAgICAgICAgaWYgcGF0aDoKICAgICAgICAgICAgc2VsZi5fbG9hZF9jc3ZfdG9fdGFibGUocGF0aCkKCiAgICBkZWYgX2xvYWRfY3N2X3RvX3RhYmxlKHNlbGYsIHBhdGgpOgogICAgICAgIHRyeToKICAgICAgICAgICAgd2l0aCBvcGVuKHBhdGgsICJyIiwgbmV3bGluZT0nJykgYXMgZjoKICAgICAgICAgICAgICAgIHJlYWRlciA9IGNzdi5yZWFkZXIoZikKICAgICAgICAgICAgICAgIGhlYWRlciA9IG5leHQocmVhZGVyLCBOb25lKQogICAgICAgICAgICAgICAgcm93cyA9IGxpc3QocmVhZGVyKQogICAgICAgICAgICAKICAgICAgICAgICAgaWYgaGVhZGVyIGFuZCBsZW4oaGVhZGVyKSA+PSAzOgogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBjb2xzID0gWyJzdWJqZWN0X2lkIiwidHJ1ZV9sYWJlbCIsInByZWRfbGFiZWwiLCJzY29yZSJdCiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0Q29sdW1uQ291bnQobGVuKGNvbHMpKQogICAgICAgICAgICAgICAgc2VsZi50YWJsZS5zZXRIb3Jpem9udGFsSGVhZGVyTGFiZWxzKGNvbHMpCiAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KGxlbihyb3dzKSkKICAgICAgICAgICAgICAgIGZvciBpLCByb3cgaW4gZW51bWVyYXRlKHJvd3MpOgogICAgICAgICAgICAgICAgICAgIGZvciBqIGluIHJhbmdlKG1pbihsZW4oY29scyksIGxlbihyb3cpKSk6CiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0SXRlbShpLCBqLCBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0SXRlbShyb3dbal0pKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KGxlbihyb3dzKSkKICAgICAgICAgICAgICAgIGZvciBpLCByb3cgaW4gZW51bWVyYXRlKHJvd3MpOgogICAgICAgICAgICAgICAgICAgIGZvciBqIGluIHJhbmdlKG1pbig0LCBsZW4ocm93KSkpOgogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldEl0ZW0oaSwgaiwgUXRXaWRnZXRzLlFUYWJsZVdpZGdldEl0ZW0ocm93W2pdKSkKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJJbXBvcnQiLCBmIkxvYWRlZDoge29zLnBhdGguYmFzZW5hbWUocGF0aCl9IikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAiSW1wb3J0IGVycm9yIiwgc3RyKGUpKQoKICAgIGRlZiBleHBvcnRfcmVzdWx0cyhzZWxmKToKICAgICAgICBlbnN1cmVfZXhwb3J0X2RpcigpCiAgICAgICAgCiAgICAgICAgaWYgbm90IHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkuc3RyaXAoKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIGRlZmF1bHQgPSBvcy5wYXRoLmFic3BhdGgob3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiYWdyZWVtZW50X3Jlc3VsdHNfe3RzfS50eHQiKSkKICAgICAgICBwYXRoLCBfID0gUXRXaWRnZXRzLlFGaWxlRGlhbG9nLmdldFNhdmVGaWxlTmFtZShzZWxmLCAiU2F2ZSBSZXN1bHRzIEFzIiwgZGVmYXVsdCwgIlRleHQgRmlsZXMgKCoudHh0KSIpCiAgICAgICAgaWYgbm90IHBhdGg6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aChwYXRoKX0iKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImljY19yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAKICAgICAgICBpZiBub3Qgc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKS5zdHJpcCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgZGVmYXVsdCA9IG9zLnBhdGguYWJzcGF0aChvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJmbGVpc3NfcmVzdWx0c197dHN9LnR4dCIpKQogICAgICAgIHBhdGgsIF8gPSBRdFdpZGdldHMuUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsICJTYXZlIFJlc3VsdHMgQXMiLCBkZWZhdWx0LCAiVGV4dCBGaWxlcyAoKi50eHQpIikKICAgICAgICBpZiBub3QgcGF0aDoKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiRXhwb3J0IFJlc3VsdHMiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHBhdGgpfSIpCiAgICAgICAgCiAgICAgICAgaWYgbm90IHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkuc3RyaXAoKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIGRlZmF1bHQgPSBvcy5wYXRoLmFic3BhdGgob3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiYmluYXJ5X3Jlc3VsdHNfe3RzfS50eHQiKSkKICAgICAgICBwYXRoLCBfID0gUXRXaWRnZXRzLlFGaWxlRGlhbG9nLmdldFNhdmVGaWxlTmFtZShzZWxmLCAiU2F2ZSBSZXN1bHRzIEFzIiwgZGVmYXVsdCwgIlRleHQgRmlsZXMgKCoudHh0KSIpCiAgICAgICAgaWYgbm90IHBhdGg6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aChwYXRoKX0iKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBfZToKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBuYW1lID0gKHNlbGYubmFtZV9lZGl0LnRleHQoKS5zdHJpcCgpIG9yICJHcm91cCIpLnJlcGxhY2UoJyAnLCAnXycpCiAgICAgICAgdHh0X3BhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJiaW5hcnlfcmVzdWx0c197bmFtZX1fe3RzfS50eHQiKQogICAgICAgIHdpdGggb3Blbih0eHRfcGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgodHh0X3BhdGgpfSIpCgogICAgCiAgICBkZWYgYWRkX3Jvd3Moc2VsZik6CiAgICAgICAgc2VsZi50YWJsZS5zZXRSb3dDb3VudChzZWxmLnRhYmxlLnJvd0NvdW50KCkrMTApCgogICAgZGVmIGFkZF9vbmVfcm93KHNlbGYpOgogICAgICAgIHNlbGYudGFibGUuc2V0Um93Q291bnQoc2VsZi50YWJsZS5yb3dDb3VudCgpKzEpCgogICAgZGVmIHJlbV9vbmVfcm93KHNlbGYpOgogICAgICAgIGlmIHNlbGYudGFibGUucm93Q291bnQoKSA+IDE6CiAgICAgICAgICAgIHNlbGYudGFibGUuc2V0Um93Q291bnQoc2VsZi50YWJsZS5yb3dDb3VudCgpLTEpCgogICAgCiAgICBkZWYgX2NvbGxlY3Qoc2VsZik6CiAgICAgICAgeV90cnVlLCB5X3ByZWQsIHNjb3JlcyA9IFtdLCBbXSwgW10KICAgICAgICBmb3IgciBpbiByYW5nZShzZWxmLnRhYmxlLnJvd0NvdW50KCkpOgogICAgICAgICAgICB0X2l0ZW0gPSBzZWxmLnRhYmxlLml0ZW0ociwxKTsgcF9pdGVtID0gc2VsZi50YWJsZS5pdGVtKHIsMikKICAgICAgICAgICAgaWYgbm90IHRfaXRlbSBvciBub3QgcF9pdGVtOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgdCA9IHRfaXRlbS50ZXh0KCkuc3RyaXAoKTsgcCA9IHBfaXRlbS50ZXh0KCkuc3RyaXAoKQogICAgICAgICAgICBpZiB0PT0iIiBvciBwPT0iIjoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHQgPSBpbnQoZmxvYXQodCkpOyBwID0gaW50KGZsb2F0KHApKQogICAgICAgICAgICBleGNlcHQ6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICB5X3RydWUuYXBwZW5kKHQpOyB5X3ByZWQuYXBwZW5kKHApCiAgICAgICAgICAgIHNfaXRlbSA9IHNlbGYudGFibGUuaXRlbShyLDMpCiAgICAgICAgICAgIGlmIHNfaXRlbSBhbmQgc19pdGVtLnRleHQoKS5zdHJpcCgpIT0iIjoKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBzY29yZXMuYXBwZW5kKGZsb2F0KHNfaXRlbS50ZXh0KCkuc3RyaXAoKSkpCiAgICAgICAgICAgICAgICBleGNlcHQ6CiAgICAgICAgICAgICAgICAgICAgc2NvcmVzLmFwcGVuZChOb25lKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgc2NvcmVzLmFwcGVuZChOb25lKQogICAgICAgIHNjb3JlX3ZhbHMgPSBbcyBmb3IgcyBpbiBzY29yZXMgaWYgaXNpbnN0YW5jZShzLChpbnQsZmxvYXQpKV0KICAgICAgICByZXR1cm4geV90cnVlLCB5X3ByZWQsIHNjb3JlX3ZhbHMgaWYgbGVuKHNjb3JlX3ZhbHMpPT1sZW4oeV90cnVlKSBlbHNlIFtdCgogICAgCiAgICBkZWYgY2FsY3VsYXRlKHNlbGYpOgogICAgICAgIHlfdHJ1ZSwgeV9wcmVkLCBzY29yZXMgPSBzZWxmLl9jb2xsZWN0KCkKICAgICAgICBpZiBub3QgeV90cnVlOgogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3gud2FybmluZyhzZWxmLCJJbnB1dCIsIk5vIHZhbGlkIHJvd3MuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgcmVzID0gYmluYXJ5X21ldHJpY3MoeV90cnVlLCB5X3ByZWQpCiAgICAgICAgbGluZXMgPSBbXQogICAgICAgIG5hbWUgPSBzZWxmLm5hbWVfZWRpdC50ZXh0KCkuc3RyaXAoKQogICAgICAgIGlmIG5hbWU6CiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIkxhYmVsOiB7bmFtZX0iKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiIpCiAgICAgICAgZm9yIGsgaW4gWyJUUCIsIkZQIiwiVE4iLCJGTiIsIk4iXToKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYie2t9OiB7cmVzW2tdfSIpCiAgICAgICAgbGluZXMuYXBwZW5kKCIiKQogICAgICAgIGRlZiBmbXRfcGN0KHgpOgogICAgICAgICAgICByZXR1cm4gIm5hbiIgaWYgKHggaXMgTm9uZSBvciAoaXNpbnN0YW5jZSh4LGZsb2F0KSBhbmQgbWF0aC5pc25hbih4KSkpIGVsc2UgZiJ7eCoxMDA6LjJmfSUiCiAgICAgICAgbGluZXMuYXBwZW5kKGYiQWNjdXJhY3k6IHtmbXRfcGN0KHJlc1snQWNjdXJhY3knXSl9ICBDSSBbe2ZtdF9wY3QocmVzWydBY2N1cmFjeV9DSSddWzBdKX0sIHtmbXRfcGN0KHJlc1snQWNjdXJhY3lfQ0knXVsxXSl9XSIpCiAgICAgICAgbGluZXMuYXBwZW5kKGYiU2Vuc2l0aXZpdHkgKFRQUik6IHtmbXRfcGN0KHJlc1snU2Vuc2l0aXZpdHlfVFBSJ10pfSAgQ0kgW3tmbXRfcGN0KHJlc1snU2Vuc2l0aXZpdHlfQ0knXVswXSl9LCB7Zm10X3BjdChyZXNbJ1NlbnNpdGl2aXR5X0NJJ11bMV0pfV0iKQogICAgICAgIGxpbmVzLmFwcGVuZChmIlNwZWNpZmljaXR5IChUTlIpOiB7Zm10X3BjdChyZXNbJ1NwZWNpZmljaXR5X1ROUiddKX0gIENJIFt7Zm10X3BjdChyZXNbJ1NwZWNpZmljaXR5X0NJJ11bMF0pfSwge2ZtdF9wY3QocmVzWydTcGVjaWZpY2l0eV9DSSddWzFdKX1dIikKICAgICAgICBsaW5lcy5hcHBlbmQoZiJQUFY6IHtmbXRfcGN0KHJlc1snUFBWJ10pfSAgQ0kgW3tmbXRfcGN0KHJlc1snUFBWX0NJJ11bMF0pfSwge2ZtdF9wY3QocmVzWydQUFZfQ0knXVsxXSl9XSIpCiAgICAgICAgbGluZXMuYXBwZW5kKGYiTlBWOiB7Zm10X3BjdChyZXNbJ05QViddKX0gIENJIFt7Zm10X3BjdChyZXNbJ05QVl9DSSddWzBdKX0sIHtmbXRfcGN0KHJlc1snTlBWX0NJJ11bMV0pfV0iKQogICAgICAgIGxpbmVzLmFwcGVuZChmIkYxOiB7Zm10X3BjdChyZXNbJ0YxJ10pfSIpCiAgICAgICAgbGluZXMuYXBwZW5kKGYiQmFsYW5jZWQgQWNjdXJhY3k6IHtmbXRfcGN0KHJlc1snQmFsYW5jZWRfQWNjdXJhY3knXSl9IikKICAgICAgICB5aiA9IHJlc1snWW91ZGVuc19KJ107IHlqX3N0ciA9ICduYW4nIGlmIG1hdGguaXNuYW4oeWopIGVsc2UgZiJ7eWo6LjRmfSIKICAgICAgICBtY2MgPSByZXNbJ01DQyddOyBtY2Nfc3RyID0gJ25hbicgaWYgbWF0aC5pc25hbihtY2MpIGVsc2UgZiJ7bWNjOi40Zn0iCiAgICAgICAgbGluZXMuYXBwZW5kKGYiWW91ZGVuJ3MgSjoge3lqX3N0cn0iKQogICAgICAgIGxpbmVzLmFwcGVuZChmIk1DQzoge21jY19zdHJ9IikKICAgICAgICBpZiBzY29yZXMgYW5kIGxlbihzZXQoc2NvcmVzKSk+MToKICAgICAgICAgICAgeHMsIHlzLCBhdWMgPSByb2NfcG9pbnRzX2F1Yyh5X3RydWUsIHNjb3JlcykKICAgICAgICAgICAgciwgcCwgYXAgPSBwcl9wb2ludHNfYXAoeV90cnVlLCBzY29yZXMpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIkFVQyAoUk9DKToge2F1YzouNGZ9IikKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYiQXZlcmFnZSBQcmVjaXNpb24gKFBSIEFVQyk6IHthcDouNGZ9IikKICAgICAgICBlbHNlOgogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIlJPQy9QUiBkaXNhYmxlZCAocHJvdmlkZSBTY29yZSBjb2x1bW4gd2l0aCDiiaUyIHVuaXF1ZSB2YWx1ZXMpLiIpCiAgICAgICAgc2VsZi5yZXN1bHQuc2V0UGxhaW5UZXh0KCJcbiIuam9pbihsaW5lcykpCiAgICAgICAgc2VsZi5sYXN0X2NhbGMgPSB7InJlc3VsdHMiOiByZXMsICJuYW1lIjogbmFtZX0KCiAgICBkZWYgZ3JhcGgoc2VsZik6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBpZiBzZWxmLmxhc3RfY2FsYyBpcyBOb25lOgogICAgICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgICAgICAgICAgaWYgc2VsZi5sYXN0X2NhbGMgaXMgTm9uZToKICAgICAgICAgICAgICAgICAgICByZXR1cm4KICAgICAgICAgICAgeV90cnVlLCB5X3ByZWQsIHNjb3JlcyA9IHNlbGYuX2NvbGxlY3QoKQogICAgICAgICAgICBpZiBzY29yZXMgYW5kIGxlbihzZXQoc2NvcmVzKSk+MToKICAgICAgICAgICAgICAgIHhzLCB5cywgYXVjID0gcm9jX3BvaW50c19hdWMoeV90cnVlLCBzY29yZXMpCiAgICAgICAgICAgICAgICBzZWxmLl9sYXN0X2dyYXBoX2tpbmQgPSAncm9jcHInCiAgICAgICAgICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCkKICAgICAgICAgICAgICAgIHBsdC5wbG90KHhzLCB5cywgbGFiZWw9ZiJBVUM9e2F1YzouM2Z9IikKICAgICAgICAgICAgICAgIHBsdC5wbG90KFswLDFdLFswLDFdLCctLScpCiAgICAgICAgICAgICAgICBwbHQueGxhYmVsKCJGYWxzZSBQb3NpdGl2ZSBSYXRlIik7IHBsdC55bGFiZWwoIlRydWUgUG9zaXRpdmUgUmF0ZSIpOyBwbHQudGl0bGUoIlJPQyBDdXJ2ZSIpCiAgICAgICAgICAgICAgICBwbHQubGVnZW5kKGxvYz0ibG93ZXIgcmlnaHQiKQogICAgICAgICAgICAgICAgc2VsZi5fc2hvd19maWcoZmlnKQogICAgICAgICAgICAgICAgciwgcCwgYXAgPSBwcl9wb2ludHNfYXAoeV90cnVlLCBzY29yZXMpCiAgICAgICAgICAgICAgICBzZWxmLl9sYXN0X2dyYXBoX2tpbmQgPSAncHInCiAgICAgICAgICAgICAgICBmaWcyID0gcGx0LmZpZ3VyZSgpCiAgICAgICAgICAgICAgICBwbHQuc3RlcChyLCBwLCB3aGVyZT0icG9zdCIsIGxhYmVsPWYiQVA9e2FwOi4zZn0iKQogICAgICAgICAgICAgICAgcGx0LnhsYWJlbCgiUmVjYWxsIik7IHBsdC55bGFiZWwoIlByZWNpc2lvbiIpOyBwbHQudGl0bGUoIlByZWNpc2lvbuKAk1JlY2FsbCBDdXJ2ZSIpCiAgICAgICAgICAgICAgICBwbHQubGVnZW5kKGxvYz0ibG93ZXIgbGVmdCIpCiAgICAgICAgICAgICAgICBzZWxmLl9zaG93X2ZpZyhmaWcyKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJST0MvUFIgdW5hdmFpbGFibGUiLCAiUk9DL1BSIGN1cnZlcyByZXF1aXJlIGEgbnVtZXJpYyAnc2NvcmUnIGNvbHVtbiB3aXRoIGF0IGxlYXN0IDIgdW5pcXVlIHZhbHVlcy4gU2hvd2luZyBjb25mdXNpb24gbWF0cml4IGluc3RlYWQuIikKICAgICAgICAgICAgICAgIHRwLCBmcCwgdG4sIGZuID0gY29uZnVzaW9uX2NvdW50cyh5X3RydWUsIHlfcHJlZCkKICAgICAgICAgICAgICAgIG1hdCA9IG5wLmFycmF5KFtbdG4sIGZwXSxbZm4sIHRwXV0pCiAgICAgICAgICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCkKICAgICAgICAgICAgICAgIHBsdC5pbXNob3cobWF0LCBpbnRlcnBvbGF0aW9uPSJuZWFyZXN0IikKICAgICAgICAgICAgICAgIGZvciAoaSxqKSwgdmFsIGluIG5wLm5kZW51bWVyYXRlKG1hdCk6CiAgICAgICAgICAgICAgICAgICAgcGx0LnRleHQoaiwgaSwgc3RyKHZhbCksIGhhPSJjZW50ZXIiLCB2YT0iY2VudGVyIikKICAgICAgICAgICAgICAgIHBsdC54dGlja3MoWzAsMV0sIFsiUHJlZCAwIiwiUHJlZCAxIl0pOyBwbHQueXRpY2tzKFswLDFdLCBbIlRydWUgMCIsIlRydWUgMSJdKQogICAgICAgICAgICAgICAgcGx0LnRpdGxlKCJDb25mdXNpb24gTWF0cml4Iik7IHBsdC54bGFiZWwoIlByZWRpY3RlZCIpOyBwbHQueWxhYmVsKCJUcnVlIikKICAgICAgICAgICAgICAgIHNlbGYuX3Nob3dfZmlnKGZpZykKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAiR3JhcGggZXJyb3IiLCBzdHIoZSkpCgogICAgZGVmIF9zaG93X2ZpZyhzZWxmLCBmaWcpOgogICAgICAgIGlmIG5vdCBoYXNhdHRyKHNlbGYsICdfZmlnX3dpbmRvd3MnKToKICAgICAgICAgICAgc2VsZi5fZmlnX3dpbmRvd3MgPSBbXQogICAgICAgIHcgPSBGaWd1cmVXaW5kb3coZmlnKQogICAgICAgIHNlbGYuX2ZpZ193aW5kb3dzLmFwcGVuZCh3KQogICAgICAgIHcuc2hvdygpCgogICAgZGVmIHNhdmVfZ3JhcGgoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIG5hbWUgPSAoc2VsZi5uYW1lX2VkaXQudGV4dCgpLnN0cmlwKCkgb3IgIkdyb3VwIikucmVwbGFjZSgnICcsICdfJykKICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiYmluYXJ5X2dyYXBoX3tuYW1lfV97dHN9LnBuZyIpCiAgICAgICAgeV90cnVlLCB5X3ByZWQsIHNjb3JlcyA9IHNlbGYuX2NvbGxlY3QoKQogICAgICAgIGlmIHNjb3JlcyBhbmQgbGVuKHNldChzY29yZXMpKT4xOgogICAgICAgICAgICB4cywgeXMsIGF1YyA9IHJvY19wb2ludHNfYXVjKHlfdHJ1ZSwgc2NvcmVzKQogICAgICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCkKICAgICAgICAgICAgcGx0LnBsb3QoeHMsIHlzLCBsYWJlbD1mIkFVQz17YXVjOi4zZn0iKQogICAgICAgICAgICBwbHQucGxvdChbMCwxXSxbMCwxXSwnLS0nKQogICAgICAgICAgICBwbHQueGxhYmVsKCJGYWxzZSBQb3NpdGl2ZSBSYXRlIik7IHBsdC55bGFiZWwoIlRydWUgUG9zaXRpdmUgUmF0ZSIpOyBwbHQudGl0bGUoIlJPQyBDdXJ2ZSIpCiAgICAgICAgICAgIHBsdC5sZWdlbmQobG9jPSJsb3dlciByaWdodCIpOyBmaWcuc2F2ZWZpZyhwYXRoKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHRwLCBmcCwgdG4sIGZuID0gY29uZnVzaW9uX2NvdW50cyh5X3RydWUsIHlfcHJlZCkKICAgICAgICAgICAgbWF0ID0gbnAuYXJyYXkoW1t0biwgZnBdLFtmbiwgdHBdXSkKICAgICAgICAgICAgZmlnID0gcGx0LmZpZ3VyZSgpCiAgICAgICAgICAgIHBsdC5pbXNob3cobWF0LCBpbnRlcnBvbGF0aW9uPSJuZWFyZXN0IikKICAgICAgICAgICAgZm9yIChpLGopLCB2YWwgaW4gbnAubmRlbnVtZXJhdGUobWF0KTogcGx0LnRleHQoaiwgaSwgc3RyKHZhbCksIGhhPSJjZW50ZXIiLCB2YT0iY2VudGVyIikKICAgICAgICAgICAgcGx0Lnh0aWNrcyhbMCwxXSwgWyJQcmVkIDAiLCJQcmVkIDEiXSk7IHBsdC55dGlja3MoWzAsMV0sIFsiVHJ1ZSAwIiwiVHJ1ZSAxIl0pCiAgICAgICAgICAgIHBsdC50aXRsZSgiQ29uZnVzaW9uIE1hdHJpeCIpOyBwbHQueGxhYmVsKCJQcmVkaWN0ZWQiKTsgcGx0LnlsYWJlbCgiVHJ1ZSIpOyBmaWcuc2F2ZWZpZyhwYXRoKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiU2F2ZWQgR3JhcGgiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHBhdGgpfSIpCgogICAgZGVmIHNhdmUoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIG5hbWUgPSAoc2VsZi5uYW1lX2VkaXQudGV4dCgpLnN0cmlwKCkgb3IgIkdyb3VwIikucmVwbGFjZSgnICcsICdfJykKICAgICAgICBjc3ZfcGF0aCA9IG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImJpbmFyeV9kYXRhX3tuYW1lfV97dHN9LmNzdiIpCiAgICAgICAgd2l0aCBvcGVuKGNzdl9wYXRoLCJ3IixuZXdsaW5lPSIiKSBhcyBmOgogICAgICAgICAgICB3cml0ZXIgPSBjc3Yud3JpdGVyKGYpCiAgICAgICAgICAgIHdyaXRlci53cml0ZXJvdyhbc2VsZi50YWJsZS5ob3Jpem9udGFsSGVhZGVySXRlbShpKS50ZXh0KCkgZm9yIGkgaW4gcmFuZ2Uoc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpKV0pCiAgICAgICAgICAgIGZvciByIGluIHJhbmdlKHNlbGYudGFibGUucm93Q291bnQoKSk6CiAgICAgICAgICAgICAgICByb3cgPSBbKHNlbGYudGFibGUuaXRlbShyLGMpLnRleHQoKSBpZiBzZWxmLnRhYmxlLml0ZW0ocixjKSBlbHNlICIiKSBmb3IgYyBpbiByYW5nZShzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkpXQogICAgICAgICAgICAgICAgaWYgYW55KGNlbGwuc3RyaXAoKSBmb3IgY2VsbCBpbiByb3cpOiB3cml0ZXIud3JpdGVyb3cocm93KQogICAgICAgIGpzb25fcGF0aCA9IG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImJpbmFyeV9yZXN1bHRzX3tuYW1lfV97dHN9Lmpzb24iKQogICAgICAgIHdpdGggb3Blbihqc29uX3BhdGgsInciKSBhcyBmOiBqc29uLmR1bXAoc2VsZi5sYXN0X2NhbGMgb3Ige30sIGYsIGluZGVudD0yKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCJTYXZlZCIsIGYiU2F2ZWQ6XG57b3MucGF0aC5hYnNwYXRoKGNzdl9wYXRoKX1cbntvcy5wYXRoLmFic3BhdGgoanNvbl9wYXRoKX0iKQoKICAgIAogICAgZGVmIGJhdGNoX2Zyb21fZm9sZGVyKHNlbGYpOgogICAgICAgIGZvbGRlciA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRFeGlzdGluZ0RpcmVjdG9yeShzZWxmLCAiU2VsZWN0IGZvbGRlciBvZiBDU1YgdGVtcGxhdGVzIikKICAgICAgICBpZiBub3QgZm9sZGVyOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBzdW1tYXJ5ID0gW10KICAgICAgICBmb3IgZm5hbWUgaW4gc29ydGVkKG9zLmxpc3RkaXIoZm9sZGVyKSk6CiAgICAgICAgICAgIGlmIG5vdCBmbmFtZS5sb3dlcigpLmVuZHN3aXRoKCcuY3N2Jyk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKGZvbGRlciwgZm5hbWUpCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHlfdHJ1ZSwgeV9wcmVkLCBzY29yZXMgPSBzZWxmLl9jb2xsZWN0X2Zyb21fY3N2KHBhdGgpCiAgICAgICAgICAgICAgICBpZiBub3QgeV90cnVlOgogICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgICAgICByZXMgPSBiaW5hcnlfbWV0cmljcyh5X3RydWUsIHlfcHJlZCkKICAgICAgICAgICAgICAgIGl0ZW0gPSB7CiAgICAgICAgICAgICAgICAgICAgJ2ZpbGUnOiBmbmFtZSwKICAgICAgICAgICAgICAgICAgICAnTic6IHJlc1snTiddLAogICAgICAgICAgICAgICAgICAgICdBY2N1cmFjeSc6IHJlc1snQWNjdXJhY3knXSwKICAgICAgICAgICAgICAgICAgICAnU2Vuc2l0aXZpdHknOiByZXNbJ1NlbnNpdGl2aXR5X1RQUiddLAogICAgICAgICAgICAgICAgICAgICdTcGVjaWZpY2l0eSc6IHJlc1snU3BlY2lmaWNpdHlfVE5SJ10sCiAgICAgICAgICAgICAgICAgICAgJ1BQVic6IHJlc1snUFBWJ10sCiAgICAgICAgICAgICAgICAgICAgJ05QVic6IHJlc1snTlBWJ10sCiAgICAgICAgICAgICAgICAgICAgJ0YxJzogcmVzWydGMSddLAogICAgICAgICAgICAgICAgICAgICdCYWxhbmNlZF9BY2N1cmFjeSc6IHJlc1snQmFsYW5jZWRfQWNjdXJhY3knXSwKICAgICAgICAgICAgICAgICAgICAnWW91ZGVuc19KJzogcmVzWydZb3VkZW5zX0onXSwKICAgICAgICAgICAgICAgICAgICAnTUNDJzogcmVzWydNQ0MnXQogICAgICAgICAgICAgICAgfQogICAgICAgICAgICAgICAgaWYgc2NvcmVzIGFuZCBsZW4oc2V0KHNjb3JlcykpPjE6CiAgICAgICAgICAgICAgICAgICAgeHMsIHlzLCBhdWMgPSByb2NfcG9pbnRzX2F1Yyh5X3RydWUsIHNjb3JlcykKICAgICAgICAgICAgICAgICAgICByLCBwLCBhcCA9IHByX3BvaW50c19hcCh5X3RydWUsIHNjb3JlcykKICAgICAgICAgICAgICAgICAgICBpdGVtWydBVUMnXSA9IGF1YwogICAgICAgICAgICAgICAgICAgIGl0ZW1bJ0FQJ10gPSBhcAogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICBpdGVtWydBVUMnXSA9IGZsb2F0KCduYW4nKTsgaXRlbVsnQVAnXSA9IGZsb2F0KCduYW4nKQogICAgICAgICAgICAgICAgc3VtbWFyeS5hcHBlbmQoaXRlbSkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwgIkJhdGNoIiwgZiJGYWlsZWQgb24ge2ZuYW1lfToge2V9IikKICAgICAgICBpZiBub3Qgc3VtbWFyeToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJCYXRjaCIsICJObyB2YWxpZCBDU1ZzIGZvdW5kLiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIHNlbGYuX3Nob3dfYmF0Y2hfc3VtbWFyeShzdW1tYXJ5LCBmb2xkZXIpCgogICAgZGVmIF9jb2xsZWN0X2Zyb21fY3N2KHNlbGYsIHBhdGgpOgogICAgICAgIHdpdGggb3BlbihwYXRoLCAncicsIG5ld2xpbmU9JycpIGFzIGY6CiAgICAgICAgICAgIHJlYWRlciA9IGNzdi5yZWFkZXIoZikKICAgICAgICAgICAgaGVhZGVyID0gbmV4dChyZWFkZXIsIE5vbmUpCiAgICAgICAgICAgIHJvd3MgPSBsaXN0KHJlYWRlcikKICAgICAgICB5X3RydWUsIHlfcHJlZCwgc2NvcmVzID0gW10sIFtdLCBbXQogICAgICAgIGZvciByb3cgaW4gcm93czoKICAgICAgICAgICAgaWYgbm90IHJvdzoKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIGlmIGhlYWRlciBhbmQgbGVuKGhlYWRlcikgPj0gMzoKICAgICAgICAgICAgICAgICAgICB0ID0gaW50KGZsb2F0KHJvd1sxXSkpOyBwID0gaW50KGZsb2F0KHJvd1syXSkpCiAgICAgICAgICAgICAgICAgICAgcyA9IGZsb2F0KHJvd1szXSkgaWYgbGVuKHJvdykgPiAzIGFuZCByb3dbM10gIT0gJycgZWxzZSBOb25lCiAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgIHQgPSBpbnQoZmxvYXQocm93WzFdKSk7IHAgPSBpbnQoZmxvYXQocm93WzJdKSkKICAgICAgICAgICAgICAgICAgICBzID0gZmxvYXQocm93WzNdKSBpZiBsZW4ocm93KSA+IDMgYW5kIHJvd1szXSAhPSAnJyBlbHNlIE5vbmUKICAgICAgICAgICAgZXhjZXB0OgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgeV90cnVlLmFwcGVuZCh0KTsgeV9wcmVkLmFwcGVuZChwKTsgc2NvcmVzLmFwcGVuZChzKQogICAgICAgIHNjb3JlX3ZhbHMgPSBbcyBmb3IgcyBpbiBzY29yZXMgaWYgaXNpbnN0YW5jZShzLChpbnQsZmxvYXQpKV0KICAgICAgICBzY29yZXMgPSBzY29yZV92YWxzIGlmIGxlbihzY29yZV92YWxzKSA9PSBsZW4oeV90cnVlKSBlbHNlIFtdCiAgICAgICAgcmV0dXJuIHlfdHJ1ZSwgeV9wcmVkLCBzY29yZXMKCiAgICBkZWYgX3Nob3dfYmF0Y2hfc3VtbWFyeShzZWxmLCBzdW1tYXJ5LCBmb2xkZXIpOgogICAgICAgIAogICAgICAgIGRsZyA9IFF0V2lkZ2V0cy5RRGlhbG9nKHNlbGYpOyBkbGcuc2V0V2luZG93VGl0bGUoIkJhdGNoIFN1bW1hcnkiKQogICAgICAgIHYgPSBRdFdpZGdldHMuUVZCb3hMYXlvdXQoZGxnKQogICAgICAgIHRhYmxlID0gUXRXaWRnZXRzLlFUYWJsZVdpZGdldChsZW4oc3VtbWFyeSksIDEyKQogICAgICAgIGhlYWRlcnMgPSBbImZpbGUiLCJOIiwiQWNjdXJhY3kiLCJTZW5zaXRpdml0eSIsIlNwZWNpZmljaXR5IiwiUFBWIiwiTlBWIiwiRjEiLCJCYWxhbmNlZF9BY2N1cmFjeSIsIllvdWRlbnNfSiIsIk1DQyIsIkFVQyJdCiAgICAgICAgdGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhoZWFkZXJzKQogICAgICAgIGZvciBpLCBpdGVtIGluIGVudW1lcmF0ZShzdW1tYXJ5KToKICAgICAgICAgICAgcm93ID0gWwogICAgICAgICAgICAgICAgaXRlbVsnZmlsZSddLCBpdGVtWydOJ10sIGl0ZW1bJ0FjY3VyYWN5J10sIGl0ZW1bJ1NlbnNpdGl2aXR5J10sIGl0ZW1bJ1NwZWNpZmljaXR5J10sCiAgICAgICAgICAgICAgICBpdGVtWydQUFYnXSwgaXRlbVsnTlBWJ10sIGl0ZW1bJ0YxJ10sIGl0ZW1bJ0JhbGFuY2VkX0FjY3VyYWN5J10sIGl0ZW1bJ1lvdWRlbnNfSiddLCBpdGVtWydNQ0MnXSwgaXRlbS5nZXQoJ0FVQycsIGZsb2F0KCduYW4nKSkKICAgICAgICAgICAgXQogICAgICAgICAgICBmb3IgaiwgdmFsIGluIGVudW1lcmF0ZShyb3cpOgogICAgICAgICAgICAgICAgaWYgaXNpbnN0YW5jZSh2YWwsIGZsb2F0KSBhbmQgbm90IChucC5pc25hbih2YWwpKSBhbmQgaGVhZGVyc1tqXSBub3QgaW4gKCJmaWxlIiwiTiIpOgogICAgICAgICAgICAgICAgICAgIHR4dCA9IGYie3ZhbDouNGZ9IgogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICB0eHQgPSBzdHIodmFsKQogICAgICAgICAgICAgICAgdGFibGUuc2V0SXRlbShpLCBqLCBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0SXRlbSh0eHQpKQogICAgICAgIHYuYWRkV2lkZ2V0KHRhYmxlKQogICAgICAgIGggPSBRdFdpZGdldHMuUUhCb3hMYXlvdXQoKQogICAgICAgIHNhdmVfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJTYXZlIFN1bW1hcnkgQ1NWIikKICAgICAgICBoLmFkZFN0cmV0Y2goMSk7IGguYWRkV2lkZ2V0KHNhdmVfYnRuKQogICAgICAgIHYuYWRkTGF5b3V0KGgpCgogICAgICAgIGRlZiBfc2F2ZSgpOgogICAgICAgICAgICBlbnN1cmVfZXhwb3J0X2RpcigpCiAgICAgICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgICAgICBvdXQgPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJiaW5hcnlfYmF0Y2hfc3VtbWFyeV97dHN9LmNzdiIpCiAgICAgICAgICAgIHdpdGggb3BlbihvdXQsICd3JywgbmV3bGluZT0nJykgYXMgZjoKICAgICAgICAgICAgICAgIHcgPSBjc3Yud3JpdGVyKGYpCiAgICAgICAgICAgICAgICB3LndyaXRlcm93KGhlYWRlcnMpCiAgICAgICAgICAgICAgICBmb3IgaSBpbiByYW5nZSh0YWJsZS5yb3dDb3VudCgpKToKICAgICAgICAgICAgICAgICAgICB3LndyaXRlcm93KFt0YWJsZS5pdGVtKGksIGopLnRleHQoKSBpZiB0YWJsZS5pdGVtKGksIGopIGVsc2UgJycgZm9yIGogaW4gcmFuZ2UodGFibGUuY29sdW1uQ291bnQoKSldKQogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oZGxnLCAiU2F2ZWQiLCBmIlNhdmVkOiB7b3V0fSIpCiAgICAgICAgc2F2ZV9idG4uY2xpY2tlZC5jb25uZWN0KF9zYXZlKQoKICAgICAgICBkbGcucmVzaXplKDEwMDAsIDUwMCkKICAgICAgICBkbGcuZXhlY18oKQoKCgoKY2xhc3MgRmxlaXNzVGFiKFF0V2lkZ2V0cy5RV2lkZ2V0KToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBwYXJlbnQ9Tm9uZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXyhwYXJlbnQpCiAgICAgICAgc2VsZi50YWJsZSA9IFF0V2lkZ2V0cy5RVGFibGVXaWRnZXQoMTAsIDUsIHNlbGYpCiAgICAgICAgc2VsZi50YWJsZS5zZXRIb3Jpem9udGFsSGVhZGVyTGFiZWxzKFsiaXRlbV9pZCIsInJhdGVyXzEiLCJyYXRlcl8yIiwicmF0ZXJfMyIsInJhdGVyXzQiXSkKICAgICAgICBzZWxmLnJlc3VsdCA9IFF0V2lkZ2V0cy5RUGxhaW5UZXh0RWRpdChzZWxmKTsgc2VsZi5yZXN1bHQuc2V0UmVhZE9ubHkoVHJ1ZSkKICAgICAgICBzZWxmLmNhbGNfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJDYWxjdWxhdGUiKQogICAgICAgIHNlbGYuZ3JhcGhfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJHcmFwaCIpCiAgICAgICAgc2VsZi5zYXZlX2dyYXBoX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiU2F2ZSBHcmFwaCIpCiAgICAgICAgc2VsZi5zYXZlX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiU2F2ZSBEYXRhL0pTT04iKQogICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJFeHBvcnQgUmVzdWx0cyAoLnR4dCkiKQogICAgICAgIHNlbGYuaW1wb3J0X2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiSW1wb3J0IENTViIpCiAgICAgICAgc2VsZi5hZGRfb25lX2l0ZW1fYnRuX2YgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCAxIGl0ZW0iKQogICAgICAgIHNlbGYucmVtX29uZV9pdGVtX2J0bl9mID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJSZW1vdmUgMSBpdGVtIikKICAgICAgICBzZWxmLmFkZF9yb3dfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJBZGQgMTAgcm93cyIpCiAgICAgICAgc2VsZi5sb2FkX3RwbCA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiTG9hZCB0ZW1wbGF0ZSIpCiAgICAgICAgc2VsZi5hZGRfcmF0ZXJfYnRuX2YgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCBSYXRlciIpCiAgICAgICAgc2VsZi5yZW1fcmF0ZXJfYnRuX2YgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIlJlbW92ZSBSYXRlciIpCiAgICAgICAgYnRuX3JvdyA9IFF0V2lkZ2V0cy5RSEJveExheW91dCgpCiAgICAgICAgZm9yIGIgaW4gW3NlbGYuY2FsY19idG4sIHNlbGYuZ3JhcGhfYnRuLCBzZWxmLnNhdmVfZ3JhcGhfYnRuLCBzZWxmLnNhdmVfYnRuLCBzZWxmLmV4cG9ydF9yZXN1bHRzX2J0biwKICAgICAgICAgICAgICAgICAgc2VsZi5pbXBvcnRfYnRuLCBzZWxmLmFkZF9vbmVfaXRlbV9idG5fZiwgc2VsZi5yZW1fb25lX2l0ZW1fYnRuX2YsIHNlbGYuYWRkX3Jvd19idG4sCiAgICAgICAgICAgICAgICAgIHNlbGYubG9hZF90cGwsIHNlbGYuYWRkX3JhdGVyX2J0bl9mLCBzZWxmLnJlbV9yYXRlcl9idG5fZl06CiAgICAgICAgICAgIGJ0bl9yb3cuYWRkV2lkZ2V0KGIpCiAgICAgICAgbGF5b3V0ID0gUXRXaWRnZXRzLlFWQm94TGF5b3V0KHNlbGYpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnRhYmxlKTsgbGF5b3V0LmFkZExheW91dChidG5fcm93KTsgbGF5b3V0LmFkZFdpZGdldChzZWxmLnJlc3VsdCkKICAgICAgICBzZWxmLmNhbGNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmNhbGN1bGF0ZSkKICAgICAgICBzZWxmLmdyYXBoX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5ncmFwaCkKICAgICAgICBzZWxmLnNhdmVfZ3JhcGhfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLnNhdmVfZ3JhcGgpCiAgICAgICAgc2VsZi5zYXZlX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5zYXZlKQogICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmV4cG9ydF9yZXN1bHRzKQogICAgICAgIHNlbGYuaW1wb3J0X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5pbXBvcnRfY3N2KQogICAgICAgIHNlbGYuYWRkX29uZV9pdGVtX2J0bl9mLmNsaWNrZWQuY29ubmVjdChzZWxmLmFkZF9vbmVfaXRlbSkKICAgICAgICBzZWxmLnJlbV9vbmVfaXRlbV9idG5fZi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5yZW1fb25lX2l0ZW0pCiAgICAgICAgc2VsZi5hZGRfcm93X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5hZGRfcm93cykKICAgICAgICBzZWxmLmxvYWRfdHBsLmNsaWNrZWQuY29ubmVjdChzZWxmLmxvYWRfdGVtcGxhdGUpCiAgICAgICAgc2VsZi5hZGRfcmF0ZXJfYnRuX2YuY2xpY2tlZC5jb25uZWN0KHNlbGYuYWRkX3JhdGVyKQogICAgICAgIHNlbGYucmVtX3JhdGVyX2J0bl9mLmNsaWNrZWQuY29ubmVjdChzZWxmLnJlbW92ZV9yYXRlcikKICAgICAgICBzZWxmLmxhc3QgPSBOb25lCgogICAgZGVmIGxvYWRfdGVtcGxhdGUoc2VsZik6CiAgICAgICAgcGF0aCA9IG9zLnBhdGguam9pbigiZXhhbXBsZXMiLCJmbGVpc3NfdGVtcGxhdGUuY3N2IikKICAgICAgICBpZiBub3Qgb3MucGF0aC5leGlzdHMocGF0aCk6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC53YXJuaW5nKHNlbGYsIlRlbXBsYXRlIiwiVGVtcGxhdGUgbm90IGZvdW5kLiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIHNlbGYuX2xvYWRfY3N2X3RvX3RhYmxlKHBhdGgpCgogICAgZGVmIGltcG9ydF9jc3Yoc2VsZik6CiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgIkltcG9ydCBDU1YiLCAiIiwgIkNTViBGaWxlcyAoKi5jc3YpIikKICAgICAgICBpZiBwYXRoOgogICAgICAgICAgICBzZWxmLl9sb2FkX2Nzdl90b190YWJsZShwYXRoKQoKICAgIGRlZiBfbG9hZF9jc3ZfdG9fdGFibGUoc2VsZiwgcGF0aCk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4ocGF0aCwiciIsIG5ld2xpbmU9JycpIGFzIGY6CiAgICAgICAgICAgICAgICByZWFkZXIgPSBjc3YucmVhZGVyKGYpOyBoZWFkZXIgPSBuZXh0KHJlYWRlciwgTm9uZSk7IHJvd3MgPSBsaXN0KHJlYWRlcikKICAgICAgICAgICAgaWYgaGVhZGVyOgogICAgICAgICAgICAgICAgc2VsZi50YWJsZS5zZXRDb2x1bW5Db3VudChsZW4oaGVhZGVyKSkKICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhoZWFkZXIpCiAgICAgICAgICAgIHNlbGYudGFibGUuc2V0Um93Q291bnQobWF4KDEwLCBsZW4ocm93cykpKQogICAgICAgICAgICBmb3IgaSxyb3cgaW4gZW51bWVyYXRlKHJvd3MpOgogICAgICAgICAgICAgICAgZm9yIGosdmFsIGluIGVudW1lcmF0ZShyb3cpOgogICAgICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0SXRlbShpLGosUXRXaWRnZXRzLlFUYWJsZVdpZGdldEl0ZW0odmFsKSkKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJJbXBvcnQiLCBmIkxvYWRlZDoge29zLnBhdGguYmFzZW5hbWUocGF0aCl9IikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAiSW1wb3J0IGVycm9yIiwgc3RyKGUpKQoKICAgIGRlZiBhZGRfcm93cyhzZWxmKToKICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KHNlbGYudGFibGUucm93Q291bnQoKSsxMCkKCiAgICBkZWYgYWRkX3JhdGVyKHNlbGYpOgogICAgICAgIGNvbHMgPSBzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkKICAgICAgICBzZWxmLnRhYmxlLmluc2VydENvbHVtbihjb2xzKQogICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckl0ZW0oY29scywgUXRXaWRnZXRzLlFUYWJsZVdpZGdldEl0ZW0oZiJyYXRlcl97Y29sc30iKSkKCiAgICBkZWYgcmVtb3ZlX3JhdGVyKHNlbGYpOgogICAgICAgIGlmIHNlbGYudGFibGUuY29sdW1uQ291bnQoKSA8PSAyOgogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIlJhdGVycyIsICJBdCBsZWFzdCBvbmUgcmF0ZXIgaXMgcmVxdWlyZWQuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgc2VsZi50YWJsZS5yZW1vdmVDb2x1bW4oc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpLTEpCgogICAgZGVmIGFkZF9vbmVfaXRlbShzZWxmKToKICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KHNlbGYudGFibGUucm93Q291bnQoKSsxKQoKICAgIGRlZiByZW1fb25lX2l0ZW0oc2VsZik6CiAgICAgICAgaWYgc2VsZi50YWJsZS5yb3dDb3VudCgpID4gMToKICAgICAgICAgICAgc2VsZi50YWJsZS5zZXRSb3dDb3VudChzZWxmLnRhYmxlLnJvd0NvdW50KCktMSkKCiAgICBkZWYgX2NvbGxlY3Qoc2VsZik6CiAgICAgICAgcm93cyA9IFtdCiAgICAgICAgZm9yIHIgaW4gcmFuZ2Uoc2VsZi50YWJsZS5yb3dDb3VudCgpKToKICAgICAgICAgICAgdmFscyA9IFtdCiAgICAgICAgICAgIHZhbGlkID0gRmFsc2UKICAgICAgICAgICAgZm9yIGMgaW4gcmFuZ2UoMSwgc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpKToKICAgICAgICAgICAgICAgIGl0ZW0gPSBzZWxmLnRhYmxlLml0ZW0ocixjKQogICAgICAgICAgICAgICAgaWYgaXRlbSBhbmQgaXRlbS50ZXh0KCkuc3RyaXAoKSE9IiI6CiAgICAgICAgICAgICAgICAgICAgdmFsaWQgPSBUcnVlCiAgICAgICAgICAgICAgICAgICAgdmFscy5hcHBlbmQoaXRlbS50ZXh0KCkuc3RyaXAoKSkKICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgdmFscy5hcHBlbmQoTm9uZSkKICAgICAgICAgICAgaWYgdmFsaWQ6IHJvd3MuYXBwZW5kKHZhbHMpCiAgICAgICAgcmV0dXJuIHJvd3MKCiAgICBkZWYgY2FsY3VsYXRlKHNlbGYpOgogICAgICAgIG1hdHJpeCA9IHNlbGYuX2NvbGxlY3QoKQogICAgICAgIGlmIG5vdCBtYXRyaXg6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC53YXJuaW5nKHNlbGYsIklucHV0IiwiTm8gdmFsaWQgcm93cy4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBrYXBwYSwgUF9pLCBtYXJnID0gZmxlaXNzX2thcHBhX2Zyb21fcmF3KG1hdHJpeCkKICAgICAgICBsb3csIGhpZ2ggPSBib290c3RyYXBfZmxlaXNzX2NpKG1hdHJpeCwgQj01MDApCiAgICAgICAgbGluZXMgPSBbXQogICAgICAgIGRlZiBwY3QoeCk6CiAgICAgICAgICAgIHJldHVybiAibmFuIiBpZiAoeCBpcyBOb25lIG9yIChpc2luc3RhbmNlKHgsZmxvYXQpIGFuZCBtYXRoLmlzbmFuKHgpKSkgZWxzZSBmInt4KjEwMDouMmZ9JSIKICAgICAgICBsaW5lcy5hcHBlbmQoZiJGbGVpc3MnIM66OiB7J25hbicgaWYgbWF0aC5pc25hbihrYXBwYSkgZWxzZSBmJ3trYXBwYTouNGZ9J30iKQogICAgICAgIGxpbmVzLmFwcGVuZChmIkJvb3RzdHJhcCA5NSUgQ0k6IFt7J25hbicgaWYgbWF0aC5pc25hbihsb3cpIGVsc2UgZid7bG93Oi40Zn0nfSwgeyduYW4nIGlmIG1hdGguaXNuYW4oaGlnaCkgZWxzZSBmJ3toaWdoOi40Zn0nfV0iKQogICAgICAgIGxpbmVzLmFwcGVuZCgiIikKICAgICAgICBQX2NsZWFuID0gW3ggZm9yIHggaW4gUF9pIGlmIG5vdCBtYXRoLmlzbmFuKHgpXQogICAgICAgIGlmIFBfY2xlYW46IGxpbmVzLmFwcGVuZChmIk1lYW4gcGVyLWl0ZW0gYWdyZWVtZW50IChQzIQpOiB7bnAubWVhbihQX2NsZWFuKTouNGZ9IikKICAgICAgICBsaW5lcy5hcHBlbmQoIkNhdGVnb3J5IHByZXZhbGVuY2U6IikKICAgICAgICBmb3IgY2F0LCBwIGluIG1hcmcuaXRlbXMoKTogbGluZXMuYXBwZW5kKGYiICB7Y2F0fToge3BjdChwKX0iKQogICAgICAgIHNlbGYucmVzdWx0LnNldFBsYWluVGV4dCgiXG4iLmpvaW4obGluZXMpKQogICAgICAgIHNlbGYubGFzdCA9IHsia2FwcGEiOiBrYXBwYSwgImNpIjogKGxvdywgaGlnaCksICJwZXJfaXRlbSI6IFBfaSwgIm1hcmdpbmFscyI6IG1hcmcsICJtYXRyaXgiOiBtYXRyaXh9CgogICAgZGVmIGdyYXBoKHNlbGYpOgogICAgICAgIGlmIHNlbGYubGFzdCBpcyBOb25lOgogICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgaWYgc2VsZi5sYXN0IGlzIE5vbmU6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIG1hcmcgPSBzZWxmLmxhc3RbIm1hcmdpbmFscyJdCiAgICAgICAgaWYgbm90IG1hcmc6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC53YXJuaW5nKHNlbGYsIkdyYXBoIiwiTm8gbWFyZ2luYWxzIHRvIHBsb3QuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgY2F0cyA9IGxpc3QobWFyZy5rZXlzKCkpOyB2YWxzID0gW21hcmdbY10gZm9yIGMgaW4gY2F0c10KICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCk7IHBsdC5iYXIoY2F0cywgdmFscyk7IHBsdC55bGltKDAsMSk7IHBsdC55bGFiZWwoIlByb3BvcnRpb24iKTsgcGx0LnRpdGxlKCJDYXRlZ29yeSBQcmV2YWxlbmNlIikKICAgICAgICBzZWxmLl9zaG93X2ZpZyhmaWcpCgogICAgZGVmIHNhdmVfZ3JhcGgoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIGlmIHNlbGYubGFzdCBpcyBOb25lOgogICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgaWYgc2VsZi5sYXN0IGlzIE5vbmU6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJmbGVpc3NfZ3JhcGhfe3RzfS5wbmciKQogICAgICAgIG1hcmcgPSBzZWxmLmxhc3QuZ2V0KCJtYXJnaW5hbHMiLCB7fSkKICAgICAgICBpZiBub3QgbWFyZzoKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwgIkdyYXBoIiwgIk5vIG1hcmdpbmFscyB0byBwbG90LiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGNhdHMgPSBsaXN0KG1hcmcua2V5cygpKTsgdmFscyA9IFttYXJnW2NdIGZvciBjIGluIGNhdHNdCiAgICAgICAgZmlnID0gcGx0LmZpZ3VyZSgpOyBwbHQuYmFyKGNhdHMsIHZhbHMpOyBwbHQueWxpbSgwLDEpOyBwbHQueWxhYmVsKCJQcm9wb3J0aW9uIik7IHBsdC50aXRsZSgiQ2F0ZWdvcnkgUHJldmFsZW5jZSIpCiAgICAgICAgZmlnLnNhdmVmaWcocGF0aCk7IFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiU2F2ZWQgR3JhcGgiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHBhdGgpfSIpCgogICAgZGVmIF9zaG93X2ZpZyhzZWxmLCBmaWcpOgogICAgICAgIGlmIG5vdCBoYXNhdHRyKHNlbGYsICdfZmlnX3dpbmRvd3MnKToKICAgICAgICAgICAgc2VsZi5fZmlnX3dpbmRvd3MgPSBbXQogICAgICAgIHcgPSBGaWd1cmVXaW5kb3coZmlnKQogICAgICAgIHNlbGYuX2ZpZ193aW5kb3dzLmFwcGVuZCh3KQogICAgICAgIHcuc2hvdygpCgogICAgZGVmIHNhdmUoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIGNzdl9wYXRoID0gb3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiZmxlaXNzX2RhdGFfe3RzfS5jc3YiKQogICAgICAgIHdpdGggb3Blbihjc3ZfcGF0aCwidyIsbmV3bGluZT0iIikgYXMgZjoKICAgICAgICAgICAgd3JpdGVyID0gY3N2LndyaXRlcihmKQogICAgICAgICAgICB3cml0ZXIud3JpdGVyb3coW3NlbGYudGFibGUuaG9yaXpvbnRhbEhlYWRlckl0ZW0oaSkudGV4dCgpIGZvciBpIGluIHJhbmdlKHNlbGYudGFibGUuY29sdW1uQ291bnQoKSldKQogICAgICAgICAgICBmb3IgciBpbiByYW5nZShzZWxmLnRhYmxlLnJvd0NvdW50KCkpOgogICAgICAgICAgICAgICAgcm93ID0gWyhzZWxmLnRhYmxlLml0ZW0ocixjKS50ZXh0KCkgaWYgc2VsZi50YWJsZS5pdGVtKHIsYykgZWxzZSAiIikgZm9yIGMgaW4gcmFuZ2Uoc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpKV0KICAgICAgICAgICAgICAgIGlmIGFueShjZWxsLnN0cmlwKCkgZm9yIGNlbGwgaW4gcm93KTogd3JpdGVyLndyaXRlcm93KHJvdykKICAgICAgICBqc29uX3BhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJmbGVpc3NfcmVzdWx0c197dHN9Lmpzb24iKQogICAgICAgIHdpdGggb3Blbihqc29uX3BhdGgsInciKSBhcyBmOiBqc29uLmR1bXAoc2VsZi5sYXN0IG9yIHt9LCBmLCBpbmRlbnQ9MikKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwiU2F2ZWQiLCBmIlNhdmVkOlxue29zLnBhdGguYWJzcGF0aChjc3ZfcGF0aCl9XG57b3MucGF0aC5hYnNwYXRoKGpzb25fcGF0aCl9IikKCiAgICBkZWYgZXhwb3J0X3Jlc3VsdHMoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImFncmVlbWVudF9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAKICAgICAgICBpZiBub3Qgc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKS5zdHJpcCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgZGVmYXVsdCA9IG9zLnBhdGguYWJzcGF0aChvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJpY2NfcmVzdWx0c197dHN9LnR4dCIpKQogICAgICAgIHBhdGgsIF8gPSBRdFdpZGdldHMuUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsICJTYXZlIFJlc3VsdHMgQXMiLCBkZWZhdWx0LCAiVGV4dCBGaWxlcyAoKi50eHQpIikKICAgICAgICBpZiBub3QgcGF0aDoKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiRXhwb3J0IFJlc3VsdHMiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHBhdGgpfSIpCiAgICAgICAgCiAgICAgICAgaWYgbm90IHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkuc3RyaXAoKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIGRlZmF1bHQgPSBvcy5wYXRoLmFic3BhdGgob3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiZmxlaXNzX3Jlc3VsdHNfe3RzfS50eHQiKSkKICAgICAgICBwYXRoLCBfID0gUXRXaWRnZXRzLlFGaWxlRGlhbG9nLmdldFNhdmVGaWxlTmFtZShzZWxmLCAiU2F2ZSBSZXN1bHRzIEFzIiwgZGVmYXVsdCwgIlRleHQgRmlsZXMgKCoudHh0KSIpCiAgICAgICAgaWYgbm90IHBhdGg6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aChwYXRoKX0iKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImJpbmFyeV9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAKICAgICAgICBpZiBub3Qgc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKS5zdHJpcCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgX2U6CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgdHh0X3BhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJmbGVpc3NfcmVzdWx0c197dHN9LnR4dCIpCiAgICAgICAgd2l0aCBvcGVuKHR4dF9wYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aCh0eHRfcGF0aCl9IikKCgoKCmNsYXNzIElDQ1RhYihRdFdpZGdldHMuUVdpZGdldCk6CiAgICBkZWYgX19pbml0X18oc2VsZiwgcGFyZW50PU5vbmUpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18ocGFyZW50KQogICAgICAgIHNlbGYudGFibGUgPSBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0KDEwLCA0LCBzZWxmKQogICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhbInN1YmplY3RfaWQiLCJyYXRlcl8xIiwicmF0ZXJfMiIsInJhdGVyXzMiXSkKICAgICAgICBzZWxmLnJlc3VsdCA9IFF0V2lkZ2V0cy5RUGxhaW5UZXh0RWRpdChzZWxmKTsgc2VsZi5yZXN1bHQuc2V0UmVhZE9ubHkoVHJ1ZSkKICAgICAgICBzZWxmLmNhbGNfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJDYWxjdWxhdGUiKQogICAgICAgIHNlbGYuZ3JhcGhfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJHcmFwaCIpCiAgICAgICAgc2VsZi5zYXZlX2dyYXBoX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiU2F2ZSBHcmFwaCIpCiAgICAgICAgc2VsZi5zYXZlX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiU2F2ZSBEYXRhL0pTT04iKQogICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJFeHBvcnQgUmVzdWx0cyAoLnR4dCkiKQogICAgICAgIHNlbGYuaW1wb3J0X2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiSW1wb3J0IENTViIpCiAgICAgICAgc2VsZi5hZGRfb25lX2l0ZW1fYnRuX2kgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCAxIGl0ZW0iKQogICAgICAgIHNlbGYucmVtX29uZV9pdGVtX2J0bl9pID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJSZW1vdmUgMSBpdGVtIikKICAgICAgICBzZWxmLmFkZF9yb3dfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJBZGQgMTAgcm93cyIpCiAgICAgICAgc2VsZi5sb2FkX3RwbCA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiTG9hZCB0ZW1wbGF0ZSIpCiAgICAgICAgc2VsZi5hZGRfcmF0ZXJfYnRuX2kgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCBSYXRlciIpCiAgICAgICAgc2VsZi5yZW1fcmF0ZXJfYnRuX2kgPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIlJlbW92ZSBSYXRlciIpCiAgICAgICAgYnRuX3JvdyA9IFF0V2lkZ2V0cy5RSEJveExheW91dCgpCiAgICAgICAgZm9yIGIgaW4gW3NlbGYuY2FsY19idG4sIHNlbGYuZ3JhcGhfYnRuLCBzZWxmLnNhdmVfZ3JhcGhfYnRuLCBzZWxmLnNhdmVfYnRuLCBzZWxmLmV4cG9ydF9yZXN1bHRzX2J0biwKICAgICAgICAgICAgICAgICAgc2VsZi5pbXBvcnRfYnRuLCBzZWxmLmFkZF9vbmVfaXRlbV9idG5faSwgc2VsZi5yZW1fb25lX2l0ZW1fYnRuX2ksIHNlbGYuYWRkX3Jvd19idG4sIHNlbGYubG9hZF90cGwsCiAgICAgICAgICAgICAgICAgIHNlbGYuYWRkX3JhdGVyX2J0bl9pLCBzZWxmLnJlbV9yYXRlcl9idG5faV06CiAgICAgICAgICAgIGJ0bl9yb3cuYWRkV2lkZ2V0KGIpCiAgICAgICAgbGF5b3V0ID0gUXRXaWRnZXRzLlFWQm94TGF5b3V0KHNlbGYpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnRhYmxlKTsgbGF5b3V0LmFkZExheW91dChidG5fcm93KTsgbGF5b3V0LmFkZFdpZGdldChzZWxmLnJlc3VsdCkKICAgICAgICBzZWxmLmNhbGNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmNhbGN1bGF0ZSkKICAgICAgICBzZWxmLmdyYXBoX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5ncmFwaCkKICAgICAgICBzZWxmLnNhdmVfZ3JhcGhfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLnNhdmVfZ3JhcGgpCiAgICAgICAgc2VsZi5zYXZlX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5zYXZlKQogICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmV4cG9ydF9yZXN1bHRzKQogICAgICAgIHNlbGYuaW1wb3J0X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5pbXBvcnRfY3N2KQogICAgICAgIHNlbGYuYWRkX29uZV9pdGVtX2J0bl9pLmNsaWNrZWQuY29ubmVjdChzZWxmLmFkZF9vbmVfaXRlbSkKICAgICAgICBzZWxmLnJlbV9vbmVfaXRlbV9idG5faS5jbGlja2VkLmNvbm5lY3Qoc2VsZi5yZW1fb25lX2l0ZW0pCiAgICAgICAgc2VsZi5hZGRfcm93X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5hZGRfcm93cykKICAgICAgICBzZWxmLmxvYWRfdHBsLmNsaWNrZWQuY29ubmVjdChzZWxmLmxvYWRfdGVtcGxhdGUpCiAgICAgICAgc2VsZi5hZGRfcmF0ZXJfYnRuX2kuY2xpY2tlZC5jb25uZWN0KHNlbGYuYWRkX3JhdGVyKQogICAgICAgIHNlbGYucmVtX3JhdGVyX2J0bl9pLmNsaWNrZWQuY29ubmVjdChzZWxmLnJlbW92ZV9yYXRlcikKICAgICAgICBzZWxmLmxhc3QgPSBOb25lCgogICAgZGVmIGxvYWRfdGVtcGxhdGUoc2VsZik6CiAgICAgICAgcGF0aCA9IG9zLnBhdGguam9pbigiZXhhbXBsZXMiLCJpY2NfdGVtcGxhdGUuY3N2IikKICAgICAgICBpZiBub3Qgb3MucGF0aC5leGlzdHMocGF0aCk6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC53YXJuaW5nKHNlbGYsIlRlbXBsYXRlIiwiVGVtcGxhdGUgbm90IGZvdW5kLiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIHNlbGYuX2xvYWRfY3N2X3RvX3RhYmxlKHBhdGgpCgogICAgZGVmIGltcG9ydF9jc3Yoc2VsZik6CiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgIkltcG9ydCBDU1YiLCAiIiwgIkNTViBGaWxlcyAoKi5jc3YpIikKICAgICAgICBpZiBwYXRoOgogICAgICAgICAgICBzZWxmLl9sb2FkX2Nzdl90b190YWJsZShwYXRoKQoKICAgIGRlZiBfbG9hZF9jc3ZfdG9fdGFibGUoc2VsZiwgcGF0aCk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInIiLCBuZXdsaW5lPScnKSBhcyBmOgogICAgICAgICAgICAgICAgcmVhZGVyID0gY3N2LnJlYWRlcihmKTsgaGVhZGVyID0gbmV4dChyZWFkZXIsIE5vbmUpOyByb3dzID0gbGlzdChyZWFkZXIpCiAgICAgICAgICAgIGlmIGhlYWRlcjoKICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0Q29sdW1uQ291bnQobGVuKGhlYWRlcikpCiAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldEhvcml6b250YWxIZWFkZXJMYWJlbHMoaGVhZGVyKQogICAgICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KG1heCgxMCwgbGVuKHJvd3MpKSkKICAgICAgICAgICAgZm9yIGksIHJvdyBpbiBlbnVtZXJhdGUocm93cyk6CiAgICAgICAgICAgICAgICBmb3IgaiwgdmFsIGluIGVudW1lcmF0ZShyb3cpOgogICAgICAgICAgICAgICAgICAgIHNlbGYudGFibGUuc2V0SXRlbShpLCBqLCBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0SXRlbSh2YWwpKQogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkltcG9ydCIsIGYiTG9hZGVkOiB7b3MucGF0aC5iYXNlbmFtZShwYXRoKX0iKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICJJbXBvcnQgZXJyb3IiLCBzdHIoZSkpCgogICAgZGVmIGFkZF9yb3dzKHNlbGYpOgogICAgICAgIHNlbGYudGFibGUuc2V0Um93Q291bnQoc2VsZi50YWJsZS5yb3dDb3VudCgpKzEwKQoKICAgIGRlZiBhZGRfcmF0ZXIoc2VsZik6CiAgICAgICAgY29scyA9IHNlbGYudGFibGUuY29sdW1uQ291bnQoKQogICAgICAgIHNlbGYudGFibGUuaW5zZXJ0Q29sdW1uKGNvbHMpCiAgICAgICAgc2VsZi50YWJsZS5zZXRIb3Jpem9udGFsSGVhZGVySXRlbShjb2xzLCBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0SXRlbShmInJhdGVyX3tjb2xzfSIpKQoKICAgIGRlZiByZW1vdmVfcmF0ZXIoc2VsZik6CiAgICAgICAgaWYgc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpIDw9IDI6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiUmF0ZXJzIiwgIkF0IGxlYXN0IG9uZSByYXRlciBpcyByZXF1aXJlZC4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnRhYmxlLnJlbW92ZUNvbHVtbihzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCktMSkKCiAgICBkZWYgYWRkX29uZV9pdGVtKHNlbGYpOgogICAgICAgIHNlbGYudGFibGUuc2V0Um93Q291bnQoc2VsZi50YWJsZS5yb3dDb3VudCgpKzEpCgogICAgZGVmIHJlbV9vbmVfaXRlbShzZWxmKToKICAgICAgICBpZiBzZWxmLnRhYmxlLnJvd0NvdW50KCkgPiAxOgogICAgICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KHNlbGYudGFibGUucm93Q291bnQoKS0xKQoKICAgIGRlZiBfY29sbGVjdChzZWxmKToKICAgICAgICByb3dzID0gW10KICAgICAgICBmb3IgciBpbiByYW5nZShzZWxmLnRhYmxlLnJvd0NvdW50KCkpOgogICAgICAgICAgICByb3d2YWxzID0gW10KICAgICAgICAgICAgdmFsaWQgPSBGYWxzZQogICAgICAgICAgICBmb3IgYyBpbiByYW5nZSgxLCBzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkpOgogICAgICAgICAgICAgICAgaXRlbSA9IHNlbGYudGFibGUuaXRlbShyLGMpCiAgICAgICAgICAgICAgICBpZiBpdGVtIGFuZCBpdGVtLnRleHQoKS5zdHJpcCgpIT0iIjoKICAgICAgICAgICAgICAgICAgICB2YWxpZCA9IFRydWUKICAgICAgICAgICAgICAgICAgICB0cnk6IHJvd3ZhbHMuYXBwZW5kKGZsb2F0KGl0ZW0udGV4dCgpLnN0cmlwKCkpKQogICAgICAgICAgICAgICAgICAgIGV4Y2VwdDogcm93dmFscy5hcHBlbmQobnAubmFuKQogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICByb3d2YWxzLmFwcGVuZChucC5uYW4pCiAgICAgICAgICAgIGlmIHZhbGlkOiByb3dzLmFwcGVuZChyb3d2YWxzKQogICAgICAgIHJldHVybiBucC5hcnJheShyb3dzLCBkdHlwZT1mbG9hdCkKCiAgICBkZWYgY2FsY3VsYXRlKHNlbGYpOgogICAgICAgIFggPSBzZWxmLl9jb2xsZWN0KCkKICAgICAgICBpZiBYLnNpemUgPT0gMDoKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwiSW5wdXQiLCJObyB2YWxpZCByb3dzLiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGljYywgY29tcHMgPSBpY2MyXzEoWCkKICAgICAgICBsb3csIGhpZ2ggPSBib290c3RyYXBfaWNjX2NpKFgsIEI9MTAwMCkKICAgICAgICBrID0gWC5zaGFwZVsxXQogICAgICAgIGljYzJrX3BvaW50ID0gaWNjX2F2ZyhpY2MsIGssIHJlZHVjZT0nbWVhbicpCiAgICAgICAgZGlzdCA9IGJvb3RzdHJhcF9pY2NfZGlzdHJpYnV0aW9uKFgsIEI9MTAwMCkKICAgICAgICBpZiBpc2luc3RhbmNlKGRpc3QsIG5wLm5kYXJyYXkpIGFuZCBkaXN0LnNpemUgPiAwOgogICAgICAgICAgICBpY2Mya19ib290ID0gaWNjX2F2ZyhkaXN0LCBrLCByZWR1Y2U9Tm9uZSkKICAgICAgICAgICAgd2l0aCBucC5lcnJzdGF0ZShhbGw9J2lnbm9yZScpOgogICAgICAgICAgICAgICAgbG93MmssIGhpZ2gyayA9IG5wLm5hbnBlcmNlbnRpbGUoaWNjMmtfYm9vdCwgWzIuNSwgOTcuNV0pCiAgICAgICAgICAgIGxvdzJrID0gZmxvYXQobG93MmspIGlmIG5wLmlzZmluaXRlKGxvdzJrKSBlbHNlIGZsb2F0KCJuYW4iKQogICAgICAgICAgICBoaWdoMmsgPSBmbG9hdChoaWdoMmspIGlmIG5wLmlzZmluaXRlKGhpZ2gyaykgZWxzZSBmbG9hdCgibmFuIikKICAgICAgICBlbHNlOgogICAgICAgICAgICBsb3cyaywgaGlnaDJrID0gZmxvYXQoIm5hbiIpLCBmbG9hdCgibmFuIikKCiAgICAgICAgbGluZXMgPSBbXQogICAgICAgIGxpbmVzLmFwcGVuZChmIklDQygyLDEpOiB7J25hbicgaWYgKGljYyBpcyBOb25lIG9yIChpc2luc3RhbmNlKGljYyxmbG9hdCkgYW5kIG1hdGguaXNuYW4oaWNjKSkpIGVsc2UgZid7aWNjOi40Zn0nfSIpCiAgICAgICAgbGluZXMuYXBwZW5kKGYiQm9vdHN0cmFwIDk1JSBDSTogW3snbmFuJyBpZiAobG93IGlzIE5vbmUgb3IgKGlzaW5zdGFuY2UobG93LGZsb2F0KSBhbmQgbWF0aC5pc25hbihsb3cpKSkgZWxzZSBmJ3tsb3c6LjRmfSd9LCAiCiAgICAgICAgICAgICAgICAgICAgIGYieyduYW4nIGlmIChoaWdoIGlzIE5vbmUgb3IgKGlzaW5zdGFuY2UoaGlnaCxmbG9hdCkgYW5kIG1hdGguaXNuYW4oaGlnaCkpKSBlbHNlIGYne2hpZ2g6LjRmfSd9XSIpCiAgICAgICAgbGluZXMuYXBwZW5kKGYiSUNDKDIse2t9KTogeyduYW4nIGlmIChpY2Mya19wb2ludCBpcyBOb25lIG9yIChpc2luc3RhbmNlKGljYzJrX3BvaW50LGZsb2F0KSBhbmQgbWF0aC5pc25hbihpY2Mya19wb2ludCkpKSBlbHNlIGYne2ljYzJrX3BvaW50Oi40Zn0nfSAgKGF2ZXJhZ2Ugb2Yge2t9IHJhdGVycykiKQogICAgICAgIGxpbmVzLmFwcGVuZChmIkJvb3RzdHJhcCA5NSUgQ0kgKGF2Zyk6IFt7J25hbicgaWYgbWF0aC5pc25hbihsb3cyaykgZWxzZSBmJ3tsb3cyazouNGZ9J30sICIKICAgICAgICAgICAgICAgICAgICAgZiJ7J25hbicgaWYgbWF0aC5pc25hbihoaWdoMmspIGVsc2UgZid7aGlnaDJrOi40Zn0nfV0iKQogICAgICAgIGxpbmVzLmFwcGVuZCgiIikKICAgICAgICBsaW5lcy5hcHBlbmQoIlZhcmlhbmNlIGNvbXBvbmVudHMgLyBNUyB0ZXJtczoiKQogICAgICAgIGZvciBrZXkgaW4gWyJNU1IiLCJNU0MiLCJNU0UiLCJTU1IiLCJTU0MiLCJTU0UiLCJuX3N1YmplY3RzIiwia19yYXRlcnMiLCJncmFuZF9tZWFuIl06CiAgICAgICAgICAgIHYgPSBjb21wcy5nZXQoa2V5LCBOb25lKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIHtrZXl9OiB7djouNGZ9IiBpZiBpc2luc3RhbmNlKHYsKGludCxmbG9hdCkpIGVsc2UgZiIgIHtrZXl9OiB7dn0iKQogICAgICAgIHNlbGYucmVzdWx0LnNldFBsYWluVGV4dCgiXG4iLmpvaW4obGluZXMpKQogICAgICAgIHNlbGYubGFzdCA9IHsiaWNjMl8xIjogaWNjLCAiY2kyXzEiOiAobG93LGhpZ2gpLCAiaWNjMl9rIjogaWNjMmtfcG9pbnQsICJjaTJfayI6IChsb3cyaywgaGlnaDJrKSwgImNvbXBvbmVudHMiOiBjb21wcywgIm1hdHJpeCI6IFgudG9saXN0KCl9CgogICAgZGVmIGdyYXBoKHNlbGYpOgogICAgICAgIGlmIHNlbGYubGFzdCBpcyBOb25lOgogICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgaWYgc2VsZi5sYXN0IGlzIE5vbmU6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIGNvbXBzID0gc2VsZi5sYXN0WyJjb21wb25lbnRzIl0KICAgICAgICBmaWcgPSBwbHQuZmlndXJlKCkKICAgICAgICB2YWxzID0gW2NvbXBzWyJNU1IiXSwgY29tcHNbIk1TQyJdLCBjb21wc1siTVNFIl1dCiAgICAgICAgbGFiZWxzID0gWyJNU1IgKHJvd3MpIiwiTVNDIChjb2xzKSIsIk1TRSAoZXJyb3IpIl0KICAgICAgICBwbHQuYmFyKGxhYmVscywgdmFscyk7IHBsdC50aXRsZSgiVmFyaWFuY2UgQ29tcG9uZW50cyAoTWVhbiBTcXVhcmVzKSIpOyBwbHQueWxhYmVsKCJWYWx1ZSIpCiAgICAgICAgc2VsZi5fc2hvd19maWcoZmlnKQoKICAgIGRlZiBzYXZlX2dyYXBoKHNlbGYpOgogICAgICAgIGVuc3VyZV9leHBvcnRfZGlyKCkKICAgICAgICBpZiBzZWxmLmxhc3QgaXMgTm9uZToKICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgIGlmIHNlbGYubGFzdCBpcyBOb25lOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiaWNjX2dyYXBoX3t0c30ucG5nIikKICAgICAgICBjb21wcyA9IHNlbGYubGFzdFsiY29tcG9uZW50cyJdCiAgICAgICAgZmlnID0gcGx0LmZpZ3VyZSgpCiAgICAgICAgdmFscyA9IFtjb21wc1siTVNSIl0sIGNvbXBzWyJNU0MiXSwgY29tcHNbIk1TRSJdXQogICAgICAgIGxhYmVscyA9IFsiTVNSIChyb3dzKSIsIk1TQyAoY29scykiLCJNU0UgKGVycm9yKSJdCiAgICAgICAgcGx0LmJhcihsYWJlbHMsIHZhbHMpOyBwbHQudGl0bGUoIlZhcmlhbmNlIENvbXBvbmVudHMgKE1lYW4gU3F1YXJlcykiKTsgcGx0LnlsYWJlbCgiVmFsdWUiKQogICAgICAgIGZpZy5zYXZlZmlnKHBhdGgpOyBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIlNhdmVkIEdyYXBoIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aChwYXRoKX0iKQoKICAgIGRlZiBfc2hvd19maWcoc2VsZiwgZmlnKToKICAgICAgICBpZiBub3QgaGFzYXR0cihzZWxmLCAnX2ZpZ193aW5kb3dzJyk6CiAgICAgICAgICAgIHNlbGYuX2ZpZ193aW5kb3dzID0gW10KICAgICAgICB3ID0gRmlndXJlV2luZG93KGZpZykKICAgICAgICBzZWxmLl9maWdfd2luZG93cy5hcHBlbmQodykKICAgICAgICB3LnNob3coKQoKICAgIGRlZiBzYXZlKHNlbGYpOgogICAgICAgIGVuc3VyZV9leHBvcnRfZGlyKCkKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBjc3ZfcGF0aCA9IG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImljY19kYXRhX3t0c30uY3N2IikKICAgICAgICB3aXRoIG9wZW4oY3N2X3BhdGgsInciLG5ld2xpbmU9IiIpIGFzIGY6CiAgICAgICAgICAgIHdyaXRlciA9IGNzdi53cml0ZXIoZikKICAgICAgICAgICAgd3JpdGVyLndyaXRlcm93KFtzZWxmLnRhYmxlLmhvcml6b250YWxIZWFkZXJJdGVtKGkpLnRleHQoKSBmb3IgaSBpbiByYW5nZShzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkpXSkKICAgICAgICAgICAgZm9yIHIgaW4gcmFuZ2Uoc2VsZi50YWJsZS5yb3dDb3VudCgpKToKICAgICAgICAgICAgICAgIHJvdyA9IFsoc2VsZi50YWJsZS5pdGVtKHIsYykudGV4dCgpIGlmIHNlbGYudGFibGUuaXRlbShyLGMpIGVsc2UgIiIpIGZvciBjIGluIHJhbmdlKHNlbGYudGFibGUuY29sdW1uQ291bnQoKSldCiAgICAgICAgICAgICAgICBpZiBhbnkoY2VsbC5zdHJpcCgpIGZvciBjZWxsIGluIHJvdyk6IHdyaXRlci53cml0ZXJvdyhyb3cpCiAgICAgICAganNvbl9wYXRoID0gb3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiaWNjX3Jlc3VsdHNfe3RzfS5qc29uIikKICAgICAgICB3aXRoIG9wZW4oanNvbl9wYXRoLCJ3IikgYXMgZjoganNvbi5kdW1wKHNlbGYubGFzdCBvciB7fSwgZiwgaW5kZW50PTIpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsIlNhdmVkIiwgZiJTYXZlZDpcbntvcy5wYXRoLmFic3BhdGgoY3N2X3BhdGgpfVxue29zLnBhdGguYWJzcGF0aChqc29uX3BhdGgpfSIpCgogICAgZGVmIGV4cG9ydF9yZXN1bHRzKHNlbGYpOgogICAgICAgIGVuc3VyZV9leHBvcnRfZGlyKCkKICAgICAgICAjIEF1dG8tY2FsYyBpZiBlbXB0eQogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImFncmVlbWVudF9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAjIEF1dG8tY2FsYyBpZiBlbXB0eQogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImljY19yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAjIEF1dG8tY2FsYyBpZiBlbXB0eQogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImZsZWlzc19yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAjIEF1dG8tY2FsYyBpZiBlbXB0eQogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImJpbmFyeV9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAjIEF1dG8tY2FsYyBpZiBlbXB0eQogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBfZToKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICB0eHRfcGF0aCA9IG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImljY19yZXN1bHRzX3t0c30udHh0IikKICAgICAgICB3aXRoIG9wZW4odHh0X3BhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiRXhwb3J0IFJlc3VsdHMiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHR4dF9wYXRoKX0iKQoKCgoKY2xhc3MgQWdyZWVtZW50VGFiKFF0V2lkZ2V0cy5RV2lkZ2V0KToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBwYXJlbnQ9Tm9uZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXyhwYXJlbnQpCiAgICAgICAgc2VsZi5tb2RlX2NvbWJvID0gUXRXaWRnZXRzLlFDb21ib0JveCgpCiAgICAgICAgc2VsZi5tb2RlX2NvbWJvLmFkZEl0ZW1zKFsiQ29oZW4ncyDOuiAoMiByYXRlcnMpIiwgIktyaXBwZW5kb3JmZidzIM6xIChub21pbmFsKSJdKQoKICAgICAgICBtb2RlX3JvdyA9IFF0V2lkZ2V0cy5RSEJveExheW91dCgpCiAgICAgICAgbW9kZV9yb3cuYWRkV2lkZ2V0KFF0V2lkZ2V0cy5RTGFiZWwoIk1vZGU6IikpCiAgICAgICAgbW9kZV9yb3cuYWRkV2lkZ2V0KHNlbGYubW9kZV9jb21ibykKICAgICAgICBtb2RlX3Jvdy5hZGRTdHJldGNoKDEpCgogICAgICAgIHNlbGYudGFibGUgPSBRdFdpZGdldHMuUVRhYmxlV2lkZ2V0KDEwLCAzLCBzZWxmKQogICAgICAgIHNlbGYudGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhbInVuaXRfaWQiLCJyYXRlcl8xIiwicmF0ZXJfMiJdKSAgIyBjYW4gYWRkIG1vcmUgZm9yIM6xCiAgICAgICAgc2VsZi5yZXN1bHQgPSBRdFdpZGdldHMuUVBsYWluVGV4dEVkaXQoc2VsZik7IHNlbGYucmVzdWx0LnNldFJlYWRPbmx5KFRydWUpCgogICAgICAgIHNlbGYuY2FsY19idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkNhbGN1bGF0ZSIpCiAgICAgICAgc2VsZi5ncmFwaF9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkdyYXBoIikKICAgICAgICBzZWxmLnNhdmVfZ3JhcGhfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJTYXZlIEdyYXBoIikKICAgICAgICBzZWxmLnNhdmVfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJTYXZlIERhdGEvSlNPTiIpCiAgICAgICAgc2VsZi5leHBvcnRfcmVzdWx0c19idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkV4cG9ydCBSZXN1bHRzICgudHh0KSIpCiAgICAgICAgc2VsZi5pbXBvcnRfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJJbXBvcnQgQ1NWIikKICAgICAgICBzZWxmLmxvYWRfdHBsID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJMb2FkIHRlbXBsYXRlIikKICAgICAgICBzZWxmLmFkZF9yYXRlcl9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCBSYXRlciIpCiAgICAgICAgc2VsZi5yZW1fcmF0ZXJfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJSZW1vdmUgUmF0ZXIiKQogICAgICAgIHNlbGYuYWRkX3Jvd19idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCAxMCByb3dzIikKICAgICAgICBzZWxmLmFkZF9vbmVfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJBZGQgMSBpdGVtIikKICAgICAgICBzZWxmLnJlbV9vbmVfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJSZW1vdmUgMSBpdGVtIikKCiAgICAgICAgYnRuX3JvdyA9IFF0V2lkZ2V0cy5RSEJveExheW91dCgpCiAgICAgICAgZm9yIGIgaW4gW3NlbGYuY2FsY19idG4sIHNlbGYuZ3JhcGhfYnRuLCBzZWxmLnNhdmVfZ3JhcGhfYnRuLCBzZWxmLnNhdmVfYnRuLCBzZWxmLmV4cG9ydF9yZXN1bHRzX2J0biwKICAgICAgICAgICAgICAgICAgc2VsZi5pbXBvcnRfYnRuLCBzZWxmLmxvYWRfdHBsLCBzZWxmLmFkZF9yYXRlcl9idG4sIHNlbGYucmVtX3JhdGVyX2J0biwgc2VsZi5hZGRfcm93X2J0biwgc2VsZi5hZGRfb25lX2J0biwgc2VsZi5yZW1fb25lX2J0bl06CiAgICAgICAgICAgIGJ0bl9yb3cuYWRkV2lkZ2V0KGIpCgogICAgICAgIGxheW91dCA9IFF0V2lkZ2V0cy5RVkJveExheW91dChzZWxmKQogICAgICAgIGxheW91dC5hZGRMYXlvdXQobW9kZV9yb3cpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnRhYmxlKQogICAgICAgIGxheW91dC5hZGRMYXlvdXQoYnRuX3JvdykKICAgICAgICBsYXlvdXQuYWRkV2lkZ2V0KHNlbGYucmVzdWx0KQoKICAgICAgICBzZWxmLmNhbGNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmNhbGN1bGF0ZSkKICAgICAgICBzZWxmLmdyYXBoX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5ncmFwaCkKICAgICAgICBzZWxmLnNhdmVfZ3JhcGhfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLnNhdmVfZ3JhcGgpCiAgICAgICAgc2VsZi5zYXZlX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5zYXZlKQogICAgICAgIHNlbGYuZXhwb3J0X3Jlc3VsdHNfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmV4cG9ydF9yZXN1bHRzKQogICAgICAgIHNlbGYuaW1wb3J0X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5pbXBvcnRfY3N2KQogICAgICAgIHNlbGYubG9hZF90cGwuY2xpY2tlZC5jb25uZWN0KHNlbGYubG9hZF90ZW1wbGF0ZSkKICAgICAgICBzZWxmLmFkZF9yYXRlcl9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuYWRkX3JhdGVyKQogICAgICAgIHNlbGYucmVtX3JhdGVyX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5yZW1vdmVfcmF0ZXIpCiAgICAgICAgc2VsZi5hZGRfcm93X2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5hZGRfcm93cykKICAgICAgICBzZWxmLmFkZF9vbmVfYnRuLmNsaWNrZWQuY29ubmVjdChzZWxmLmFkZF9vbmUpCiAgICAgICAgc2VsZi5yZW1fb25lX2J0bi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5yZW1fb25lKQogICAgICAgIHNlbGYubGFzdCA9IE5vbmUKCiAgICBkZWYgYWRkX3Jvd3Moc2VsZik6CiAgICAgICAgc2VsZi50YWJsZS5zZXRSb3dDb3VudChzZWxmLnRhYmxlLnJvd0NvdW50KCkrMTApCiAgICBkZWYgYWRkX29uZShzZWxmKToKICAgICAgICBzZWxmLnRhYmxlLnNldFJvd0NvdW50KHNlbGYudGFibGUucm93Q291bnQoKSsxKQogICAgZGVmIHJlbV9vbmUoc2VsZik6CiAgICAgICAgaWYgc2VsZi50YWJsZS5yb3dDb3VudCgpPjE6IHNlbGYudGFibGUuc2V0Um93Q291bnQoc2VsZi50YWJsZS5yb3dDb3VudCgpLTEpCgogICAgZGVmIGFkZF9yYXRlcihzZWxmKToKICAgICAgICBjb2xzID0gc2VsZi50YWJsZS5jb2x1bW5Db3VudCgpCiAgICAgICAgc2VsZi50YWJsZS5pbnNlcnRDb2x1bW4oY29scykKICAgICAgICBzZWxmLnRhYmxlLnNldEhvcml6b250YWxIZWFkZXJJdGVtKGNvbHMsIFF0V2lkZ2V0cy5RVGFibGVXaWRnZXRJdGVtKGYicmF0ZXJfe2NvbHN9IikpCgogICAgZGVmIHJlbW92ZV9yYXRlcihzZWxmKToKICAgICAgICBpZiBzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkgPD0gMjoKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJSYXRlcnMiLCAiTmVlZCBhdCBsZWFzdCB0d28gcmF0ZXIgY29sdW1ucy4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLnRhYmxlLnJlbW92ZUNvbHVtbihzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCktMSkKCiAgICBkZWYgbG9hZF90ZW1wbGF0ZShzZWxmKToKICAgICAgICAKICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKCJleGFtcGxlcyIsImFncmVlbWVudF90ZW1wbGF0ZS5jc3YiKQogICAgICAgIGlmIG5vdCBvcy5wYXRoLmV4aXN0cyhwYXRoKToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwgIlRlbXBsYXRlIiwgIlRlbXBsYXRlIG5vdCBmb3VuZC4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBzZWxmLl9sb2FkX2Nzdl90b190YWJsZShwYXRoKQoKICAgIGRlZiBpbXBvcnRfY3N2KHNlbGYpOgogICAgICAgIHBhdGgsIF8gPSBRdFdpZGdldHMuUUZpbGVEaWFsb2cuZ2V0T3BlbkZpbGVOYW1lKHNlbGYsICJJbXBvcnQgQ1NWIiwgIiIsICJDU1YgRmlsZXMgKCouY3N2KSIpCiAgICAgICAgaWYgcGF0aDoKICAgICAgICAgICAgc2VsZi5fbG9hZF9jc3ZfdG9fdGFibGUocGF0aCkKCiAgICBkZWYgX2xvYWRfY3N2X3RvX3RhYmxlKHNlbGYsIHBhdGgpOgogICAgICAgIHRyeToKICAgICAgICAgICAgd2l0aCBvcGVuKHBhdGgsICdyJywgbmV3bGluZT0nJykgYXMgZjoKICAgICAgICAgICAgICAgIHJlYWRlciA9IGNzdi5yZWFkZXIoZikKICAgICAgICAgICAgICAgIGhlYWRlciA9IG5leHQocmVhZGVyLCBOb25lKQogICAgICAgICAgICAgICAgcm93cyA9IGxpc3QocmVhZGVyKQogICAgICAgICAgICBpZiBoZWFkZXI6CiAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldENvbHVtbkNvdW50KGxlbihoZWFkZXIpKQogICAgICAgICAgICAgICAgc2VsZi50YWJsZS5zZXRIb3Jpem9udGFsSGVhZGVyTGFiZWxzKGhlYWRlcikKICAgICAgICAgICAgc2VsZi50YWJsZS5zZXRSb3dDb3VudChtYXgoMTAsIGxlbihyb3dzKSkpCiAgICAgICAgICAgIGZvciBpLCByb3cgaW4gZW51bWVyYXRlKHJvd3MpOgogICAgICAgICAgICAgICAgZm9yIGosIHZhbCBpbiBlbnVtZXJhdGUocm93KToKICAgICAgICAgICAgICAgICAgICBzZWxmLnRhYmxlLnNldEl0ZW0oaSwgaiwgUXRXaWRnZXRzLlFUYWJsZVdpZGdldEl0ZW0odmFsKSkKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJJbXBvcnQiLCBmIkxvYWRlZDoge29zLnBhdGguYmFzZW5hbWUocGF0aCl9IikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAiSW1wb3J0IGVycm9yIiwgc3RyKGUpKQoKICAgIGRlZiBfY29sbGVjdChzZWxmKToKICAgICAgICAKICAgICAgICBkYXRhID0gW10KICAgICAgICBmb3IgciBpbiByYW5nZShzZWxmLnRhYmxlLnJvd0NvdW50KCkpOgogICAgICAgICAgICB2YWxzID0gW10KICAgICAgICAgICAgdmFsaWQgPSBGYWxzZQogICAgICAgICAgICBmb3IgYyBpbiByYW5nZSgxLCBzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkpOgogICAgICAgICAgICAgICAgaXRlbSA9IHNlbGYudGFibGUuaXRlbShyLCBjKQogICAgICAgICAgICAgICAgaWYgaXRlbSBhbmQgaXRlbS50ZXh0KCkuc3RyaXAoKSAhPSAiIjoKICAgICAgICAgICAgICAgICAgICB2YWxpZCA9IFRydWUKICAgICAgICAgICAgICAgICAgICB2YWxzLmFwcGVuZChpdGVtLnRleHQoKS5zdHJpcCgpKQogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICB2YWxzLmFwcGVuZChOb25lKQogICAgICAgICAgICBpZiB2YWxpZDoKICAgICAgICAgICAgICAgIGRhdGEuYXBwZW5kKHZhbHMpCiAgICAgICAgcmV0dXJuIGRhdGEKCiAgICBkZWYgY2FsY3VsYXRlKHNlbGYpOgogICAgICAgIGRhdGEgPSBzZWxmLl9jb2xsZWN0KCkKICAgICAgICBpZiBub3QgZGF0YToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwgIklucHV0IiwgIk5vIHZhbGlkIHJvd3MuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgbW9kZSA9IHNlbGYubW9kZV9jb21iby5jdXJyZW50VGV4dCgpCiAgICAgICAgbGluZXMgPSBbXQogICAgICAgIGlmICJDb2hlbiIgaW4gbW9kZToKICAgICAgICAgICAgCiAgICAgICAgICAgIGlmIGxlbihkYXRhWzBdKSAhPSAyOgogICAgICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94Lndhcm5pbmcoc2VsZiwgIkNvaGVuJ3MgzroiLCAiRXhhY3RseSAyIHJhdGVyIGNvbHVtbnMgcmVxdWlyZWQuIikKICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICByMSA9IFtyb3dbMF0gZm9yIHJvdyBpbiBkYXRhXQogICAgICAgICAgICByMiA9IFtyb3dbMV0gZm9yIHJvdyBpbiBkYXRhXQogICAgICAgICAgICBrYXBwYSwgaW5mbyA9IGNvaGVuc19rYXBwYShyMSwgcjIpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIkNvaGVuJ3Mgzro6IHsnbmFuJyBpZiAoaXNpbnN0YW5jZShrYXBwYSxmbG9hdCkgYW5kIG1hdGguaXNuYW4oa2FwcGEpKSBlbHNlIGYne2thcHBhOi40Zn0nfSIpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIk9ic2VydmVkIGFncmVlbWVudCAoUG8pOiB7J25hbicgaWYgKGlzaW5zdGFuY2UoaW5mb1sncG8nXSxmbG9hdCkgYW5kIG1hdGguaXNuYW4oaW5mb1sncG8nXSkpIGVsc2UgZid7aW5mb1sncG8nXTouNGZ9J30iKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoZiJFeHBlY3RlZCBhZ3JlZW1lbnQgKFBlKTogeyduYW4nIGlmIChpc2luc3RhbmNlKGluZm9bJ3BlJ10sZmxvYXQpIGFuZCBtYXRoLmlzbmFuKGluZm9bJ3BlJ10pKSBlbHNlIGYne2luZm9bJ3BlJ106LjRmfSd9IikKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYiTjoge2luZm9bJ24nXX0iKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiIpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZCgiQ2F0ZWdvcnkgZnJlcXVlbmNpZXM6IikKICAgICAgICAgICAgZm9yIGssIHYgaW4gaW5mb1snY2F0ZWdvcmllcyddLml0ZW1zKCk6CiAgICAgICAgICAgICAgICBsaW5lcy5hcHBlbmQoZiIgIHtrfToge3Z9IikKICAgICAgICAgICAgc2VsZi5sYXN0ID0geyJtb2RlIjogImNvaGVuIiwgImthcHBhIjoga2FwcGEsICJpbmZvIjogaW5mb30KICAgICAgICBlbHNlOgogICAgICAgICAgICBhbHBoYSwgaW5mbyA9IGtyaXBwZW5kb3JmZl9hbHBoYV9ub21pbmFsKGRhdGEpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIktyaXBwZW5kb3JmZidzIM6xIChub21pbmFsKTogeyduYW4nIGlmIChpc2luc3RhbmNlKGFscGhhLGZsb2F0KSBhbmQgbWF0aC5pc25hbihhbHBoYSkpIGVsc2UgZid7YWxwaGE6LjRmfSd9IikKICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYiRG8gKG9ic2VydmVkIGRpc2FncmVlbWVudCk6IHsnbmFuJyBpZiAoaXNpbnN0YW5jZShpbmZvWydEbyddLGZsb2F0KSBhbmQgbWF0aC5pc25hbihpbmZvWydEbyddKSkgZWxzZSBmJ3tpbmZvWydEbyddOi40Zn0nfSIpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIkRlIChleHBlY3RlZCBkaXNhZ3JlZW1lbnQpOiB7J25hbicgaWYgKGlzaW5zdGFuY2UoaW5mb1snRGUnXSxmbG9hdCkgYW5kIG1hdGguaXNuYW4oaW5mb1snRGUnXSkpIGVsc2UgZid7aW5mb1snRGUnXTouNGZ9J30iKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoZiJUb3RhbCBwYWlycyBjb25zaWRlcmVkOiB7aW5mb1sndG90YWxfcGFpcnMnXX0iKQogICAgICAgICAgICBsaW5lcy5hcHBlbmQoIiIpCiAgICAgICAgICAgIGxpbmVzLmFwcGVuZCgiQ2F0ZWdvcnkgZnJlcXVlbmNpZXM6IikKICAgICAgICAgICAgZm9yIGssIHYgaW4gc29ydGVkKGluZm9bJ2NhdGVnb3J5X2ZyZXF1ZW5jeSddLml0ZW1zKCkpOgogICAgICAgICAgICAgICAgbGluZXMuYXBwZW5kKGYiICB7a306IHt2fSIpCiAgICAgICAgICAgIHNlbGYubGFzdCA9IHsibW9kZSI6ICJrcmlwcGVuZG9yZmYiLCAiYWxwaGEiOiBhbHBoYSwgImluZm8iOiBpbmZvfQogICAgICAgIHNlbGYucmVzdWx0LnNldFBsYWluVGV4dCgiXG4iLmpvaW4obGluZXMpKQoKICAgIGRlZiBncmFwaChzZWxmKToKICAgICAgICBpZiBzZWxmLmxhc3QgaXMgTm9uZToKICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgIGlmIHNlbGYubGFzdCBpcyBOb25lOgogICAgICAgICAgICByZXR1cm4KICAgICAgICAKICAgICAgICBpbmZvID0gc2VsZi5sYXN0WydpbmZvJ10KICAgICAgICBjYXRzID0gW10KICAgICAgICB2YWxzID0gW10KICAgICAgICBpZiBzZWxmLmxhc3RbJ21vZGUnXSA9PSAnY29oZW4nOgogICAgICAgICAgICBmcmVxID0gaW5mb1snY2F0ZWdvcmllcyddCiAgICAgICAgZWxzZToKICAgICAgICAgICAgZnJlcSA9IGluZm9bJ2NhdGVnb3J5X2ZyZXF1ZW5jeSddCiAgICAgICAgaWYgbm90IGZyZXE6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiR3JhcGgiLCAiTm8gY2F0ZWdvcmllcyB0byBwbG90LiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIGZvciBrLCB2IGluIHNvcnRlZChmcmVxLml0ZW1zKCkpOgogICAgICAgICAgICBjYXRzLmFwcGVuZChzdHIoaykpCiAgICAgICAgICAgIHZhbHMuYXBwZW5kKHYpCiAgICAgICAgcyA9IHN1bSh2YWxzKQogICAgICAgIHByb3BzID0gW3YvcyBpZiBzIGVsc2UgMCBmb3IgdiBpbiB2YWxzXQogICAgICAgIGZpZyA9IHBsdC5maWd1cmUoKQogICAgICAgIHBsdC5iYXIoY2F0cywgcHJvcHMpCiAgICAgICAgcGx0LnlsaW0oMCwgMSkKICAgICAgICBwbHQueWxhYmVsKCJQcm9wb3J0aW9uIikKICAgICAgICBwbHQudGl0bGUoIkNhdGVnb3J5IFByZXZhbGVuY2UiKQogICAgICAgIHNlbGYuX3Nob3dfZmlnKGZpZykKCiAgICBkZWYgX3Nob3dfZmlnKHNlbGYsIGZpZyk6CiAgICAgICAgaWYgbm90IGhhc2F0dHIoc2VsZiwgJ19maWdfd2luZG93cycpOgogICAgICAgICAgICBzZWxmLl9maWdfd2luZG93cyA9IFtdCiAgICAgICAgdyA9IEZpZ3VyZVdpbmRvdyhmaWcpCiAgICAgICAgc2VsZi5fZmlnX3dpbmRvd3MuYXBwZW5kKHcpCiAgICAgICAgdy5zaG93KCkKCiAgICBkZWYgc2F2ZV9ncmFwaChzZWxmKToKICAgICAgICBlbnN1cmVfZXhwb3J0X2RpcigpCiAgICAgICAgaWYgc2VsZi5sYXN0IGlzIE5vbmU6CiAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICBpZiBzZWxmLmxhc3QgaXMgTm9uZToKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgZm5hbWUgPSAnY29oZW4nIGlmIHNlbGYubGFzdFsnbW9kZSddID09ICdjb2hlbicgZWxzZSAna3JpcHBlbmRvcmZmJwogICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJhZ3JlZW1lbnRfe2ZuYW1lfV9ncmFwaF97dHN9LnBuZyIpCiAgICAgICAgaW5mbyA9IHNlbGYubGFzdFsnaW5mbyddCiAgICAgICAgZnJlcSA9IGluZm9bJ2NhdGVnb3JpZXMnXSBpZiBzZWxmLmxhc3RbJ21vZGUnXSA9PSAnY29oZW4nIGVsc2UgaW5mb1snY2F0ZWdvcnlfZnJlcXVlbmN5J10KICAgICAgICBjYXRzLCB2YWxzID0gemlwKCpzb3J0ZWQoZnJlcS5pdGVtcygpKSkgaWYgZnJlcSBlbHNlIChbXSwgW10pCiAgICAgICAgcyA9IHN1bSh2YWxzKSBpZiB2YWxzIGVsc2UgMAogICAgICAgIHByb3BzID0gW3YvcyBpZiBzIGVsc2UgMCBmb3IgdiBpbiB2YWxzXQogICAgICAgIGZpZyA9IHBsdC5maWd1cmUoKTsgcGx0LmJhcihjYXRzLCBwcm9wcyk7IHBsdC55bGltKDAsMSk7IHBsdC55bGFiZWwoIlByb3BvcnRpb24iKTsgcGx0LnRpdGxlKCJDYXRlZ29yeSBQcmV2YWxlbmNlIikKICAgICAgICBmaWcuc2F2ZWZpZyhwYXRoKTsgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJTYXZlZCBHcmFwaCIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKCiAgICBkZWYgc2F2ZShzZWxmKToKICAgICAgICBlbnN1cmVfZXhwb3J0X2RpcigpCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgY3N2X3BhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJhZ3JlZW1lbnRfZGF0YV97dHN9LmNzdiIpCiAgICAgICAgd2l0aCBvcGVuKGNzdl9wYXRoLCAidyIsIG5ld2xpbmU9IiIpIGFzIGY6CiAgICAgICAgICAgIHcgPSBjc3Yud3JpdGVyKGYpCiAgICAgICAgICAgIHcud3JpdGVyb3coW3NlbGYudGFibGUuaG9yaXpvbnRhbEhlYWRlckl0ZW0oaSkudGV4dCgpIGZvciBpIGluIHJhbmdlKHNlbGYudGFibGUuY29sdW1uQ291bnQoKSldKQogICAgICAgICAgICBmb3IgciBpbiByYW5nZShzZWxmLnRhYmxlLnJvd0NvdW50KCkpOgogICAgICAgICAgICAgICAgcm93ID0gWyhzZWxmLnRhYmxlLml0ZW0ociwgYykudGV4dCgpIGlmIHNlbGYudGFibGUuaXRlbShyLCBjKSBlbHNlICIiKSBmb3IgYyBpbiByYW5nZShzZWxmLnRhYmxlLmNvbHVtbkNvdW50KCkpXQogICAgICAgICAgICAgICAgaWYgYW55KGNlbGwuc3RyaXAoKSBmb3IgY2VsbCBpbiByb3cpOgogICAgICAgICAgICAgICAgICAgIHcud3JpdGVyb3cocm93KQogICAgICAgIGpzb25fcGF0aCA9IG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImFncmVlbWVudF9yZXN1bHRzX3t0c30uanNvbiIpCiAgICAgICAgd2l0aCBvcGVuKGpzb25fcGF0aCwgJ3cnKSBhcyBmOgogICAgICAgICAgICBqc29uLmR1bXAoc2VsZi5sYXN0IG9yIHt9LCBmLCBpbmRlbnQ9MikKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIlNhdmVkIiwgZiJTYXZlZDpcbntjc3ZfcGF0aH1cbntqc29uX3BhdGh9IikKCiAgICBkZWYgZXhwb3J0X3Jlc3VsdHMoc2VsZik6CiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImFncmVlbWVudF9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAKICAgICAgICBpZiBub3Qgc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKS5zdHJpcCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgZGVmYXVsdCA9IG9zLnBhdGguYWJzcGF0aChvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJpY2NfcmVzdWx0c197dHN9LnR4dCIpKQogICAgICAgIHBhdGgsIF8gPSBRdFdpZGdldHMuUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsICJTYXZlIFJlc3VsdHMgQXMiLCBkZWZhdWx0LCAiVGV4dCBGaWxlcyAoKi50eHQpIikKICAgICAgICBpZiBub3QgcGF0aDoKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgd2l0aCBvcGVuKHBhdGgsICJ3IiwgZW5jb2Rpbmc9InV0Zi04IikgYXMgZjoKICAgICAgICAgICAgZi53cml0ZShzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpKQogICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiRXhwb3J0IFJlc3VsdHMiLCBmIlNhdmVkOiB7b3MucGF0aC5hYnNwYXRoKHBhdGgpfSIpCiAgICAgICAgCiAgICAgICAgaWYgbm90IHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkuc3RyaXAoKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5jYWxjdWxhdGUoKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIHRzID0gZGF0ZXRpbWUuZGF0ZXRpbWUubm93KCkuc3RyZnRpbWUoIiVZJW0lZF8lSCVNJVMiKQogICAgICAgIGRlZmF1bHQgPSBvcy5wYXRoLmFic3BhdGgob3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiZmxlaXNzX3Jlc3VsdHNfe3RzfS50eHQiKSkKICAgICAgICBwYXRoLCBfID0gUXRXaWRnZXRzLlFGaWxlRGlhbG9nLmdldFNhdmVGaWxlTmFtZShzZWxmLCAiU2F2ZSBSZXN1bHRzIEFzIiwgZGVmYXVsdCwgIlRleHQgRmlsZXMgKCoudHh0KSIpCiAgICAgICAgaWYgbm90IHBhdGg6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIHdpdGggb3BlbihwYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aChwYXRoKX0iKQogICAgICAgIAogICAgICAgIGlmIG5vdCBzZWxmLnJlc3VsdC50b1BsYWluVGV4dCgpLnN0cmlwKCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYuY2FsY3VsYXRlKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBkZWZhdWx0ID0gb3MucGF0aC5hYnNwYXRoKG9zLnBhdGguam9pbihFWFBPUlRfRElSLCBmImJpbmFyeV9yZXN1bHRzX3t0c30udHh0IikpCiAgICAgICAgcGF0aCwgXyA9IFF0V2lkZ2V0cy5RRmlsZURpYWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgIlNhdmUgUmVzdWx0cyBBcyIsIGRlZmF1bHQsICJUZXh0IEZpbGVzICgqLnR4dCkiKQogICAgICAgIGlmIG5vdCBwYXRoOgogICAgICAgICAgICByZXR1cm4KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciLCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOgogICAgICAgICAgICBmLndyaXRlKHNlbGYucmVzdWx0LnRvUGxhaW5UZXh0KCkpCiAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsICJFeHBvcnQgUmVzdWx0cyIsIGYiU2F2ZWQ6IHtvcy5wYXRoLmFic3BhdGgocGF0aCl9IikKICAgICAgICAKICAgICAgICBpZiBub3Qgc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKS5zdHJpcCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBzZWxmLmNhbGN1bGF0ZSgpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgX2U6CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgdHMgPSBkYXRldGltZS5kYXRldGltZS5ub3coKS5zdHJmdGltZSgiJVklbSVkXyVIJU0lUyIpCiAgICAgICAgdHh0X3BhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgZiJhZ3JlZW1lbnRfcmVzdWx0c197dHN9LnR4dCIpCiAgICAgICAgd2l0aCBvcGVuKHR4dF9wYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6CiAgICAgICAgICAgIGYud3JpdGUoc2VsZi5yZXN1bHQudG9QbGFpblRleHQoKSkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkV4cG9ydCBSZXN1bHRzIiwgZiJTYXZlZDoge29zLnBhdGguYWJzcGF0aCh0eHRfcGF0aCl9IikKCgoKCmNsYXNzIEZpZ3VyZVdpbmRvdyhRdFdpZGdldHMuUU1haW5XaW5kb3cpOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGZpZyk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5zZXRXaW5kb3dUaXRsZSgiQ2hhcnQiKQogICAgICAgIGNhbnZhcyA9IEZpZ3VyZUNhbnZhcyhmaWcpCiAgICAgICAgc2VsZi5zZXRDZW50cmFsV2lkZ2V0KGNhbnZhcykKCgoKCmNsYXNzIE1haW5XaW5kb3coUXRXaWRnZXRzLlFNYWluV2luZG93KToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCB1c2VyKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLnNldFdpbmRvd1RpdGxlKEFQUF9USVRMRSkKICAgICAgICBzZWxmLnVzZXIgPSB1c2VyCiAgICAgICAgY2VudHJhbCA9IFF0V2lkZ2V0cy5RV2lkZ2V0KHNlbGYpOyBzZWxmLnNldENlbnRyYWxXaWRnZXQoY2VudHJhbCkKICAgICAgICBsYXlvdXQgPSBRdFdpZGdldHMuUVZCb3hMYXlvdXQoY2VudHJhbCkKCiAgICAgICAgCiAgICAgICAgdG9wYmFyID0gUXRXaWRnZXRzLlFIQm94TGF5b3V0KCkKICAgICAgICB0b3BiYXIuYWRkV2lkZ2V0KFF0V2lkZ2V0cy5RTGFiZWwoZiJTaWduZWQgaW4gYXM6IHt1c2VyfSIpKQogICAgICAgIHRvcGJhci5hZGRTdHJldGNoKDEpCiAgICAgICAgdG9wYmFyLmFkZFdpZGdldChRdFdpZGdldHMuUUxhYmVsKCJEYXRlOiIpKQogICAgICAgIHNlbGYuZGF0ZV9lZGl0ID0gUXRXaWRnZXRzLlFEYXRlRWRpdChRdENvcmUuUURhdGUuY3VycmVudERhdGUoKSkKICAgICAgICBzZWxmLmRhdGVfZWRpdC5zZXRDYWxlbmRhclBvcHVwKFRydWUpCiAgICAgICAgdG9wYmFyLmFkZFdpZGdldChzZWxmLmRhdGVfZWRpdCkKCiAgICAgICAgc2VsZi5ncm91cF9jb21ibyA9IFF0V2lkZ2V0cy5RQ29tYm9Cb3goKTsgc2VsZi5ncm91cF9jb21iby5hZGRJdGVtKCJHcm91cC0xIikKICAgICAgICBzZWxmLmFkZF9ncm91cF9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkFkZCBHcm91cCIpCiAgICAgICAgc2VsZi5yZW1fZ3JvdXBfYnRuID0gUXRXaWRnZXRzLlFQdXNoQnV0dG9uKCJSZW1vdmUgR3JvdXAiKQogICAgICAgIHNlbGYucmVuYW1lX2dyb3VwX2J0biA9IFF0V2lkZ2V0cy5RUHVzaEJ1dHRvbigiUmVuYW1lIEdyb3VwIikKICAgICAgICB0b3BiYXIuYWRkU3RyZXRjaCgxKQogICAgICAgIHRvcGJhci5hZGRXaWRnZXQoUXRXaWRnZXRzLlFMYWJlbCgiR3JvdXA6IikpCiAgICAgICAgdG9wYmFyLmFkZFdpZGdldChzZWxmLmdyb3VwX2NvbWJvKQogICAgICAgIHRvcGJhci5hZGRXaWRnZXQoc2VsZi5hZGRfZ3JvdXBfYnRuKTsgdG9wYmFyLmFkZFdpZGdldChzZWxmLnJlbV9ncm91cF9idG4pOyB0b3BiYXIuYWRkV2lkZ2V0KHNlbGYucmVuYW1lX2dyb3VwX2J0bikKICAgICAgICBsYXlvdXQuYWRkTGF5b3V0KHRvcGJhcikKCiAgICAgICAgCiAgICAgICAgc2VsZi50YWJzID0gUXRXaWRnZXRzLlFUYWJXaWRnZXQoKQogICAgICAgIHNlbGYuYmluYXJ5ID0gQmluYXJ5VGFiKCk7IHNlbGYuZmxlaXNzID0gRmxlaXNzVGFiKCk7IHNlbGYuaWNjID0gSUNDVGFiKCk7IHNlbGYuYWdyZWUgPSBBZ3JlZW1lbnRUYWIoKQogICAgICAgIHNlbGYudGFicy5hZGRUYWIoc2VsZi5iaW5hcnksICJCaW5hcnkgQ2xhc3NpZmljYXRpb24iKQogICAgICAgIHNlbGYudGFicy5hZGRUYWIoc2VsZi5mbGVpc3MsICJGbGVpc3MnIM66IChBZ3JlZW1lbnQpIikKICAgICAgICBzZWxmLnRhYnMuYWRkVGFiKHNlbGYuaWNjLCAiSUNDIChSZWxpYWJpbGl0eSkiKQogICAgICAgIHNlbGYudGFicy5hZGRUYWIoc2VsZi5hZ3JlZSwgIkNvaGVuJ3MgzrogJiBLcmlwcGVuZG9yZmYncyDOsSIpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnRhYnMpCgogICAgICAgIAogICAgICAgIHNlbGYuZ3JvdXBfc3RhdGVzID0geyJHcm91cC0xIjogc2VsZi5fc25hcHNob3QoKX0KICAgICAgICBzZWxmLmdyb3VwX2NvbWJvLmN1cnJlbnRUZXh0Q2hhbmdlZC5jb25uZWN0KHNlbGYuX29uX2dyb3VwX2NoYW5nZSkKICAgICAgICBzZWxmLmFkZF9ncm91cF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuX2FkZF9ncm91cCkKICAgICAgICBzZWxmLnJlbV9ncm91cF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuX3JlbW92ZV9ncm91cCkKICAgICAgICBzZWxmLnJlbmFtZV9ncm91cF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuX3JlbmFtZV9ncm91cCkKCiAgICAgICAgCiAgICAgICAgZm9vdGVyID0gUXRXaWRnZXRzLlFIQm94TGF5b3V0KCkKICAgICAgICBzZWxmLnByZXZfc2Vzc2lvbnNfY29tYm8gPSBRdFdpZGdldHMuUUNvbWJvQm94KCkKICAgICAgICBzZWxmLmxvYWRfc2Vzc2lvbl9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIkxvYWQgU2Vzc2lvbiIpCiAgICAgICAgc2VsZi5zYXZlX2FsbF9idG4gPSBRdFdpZGdldHMuUVB1c2hCdXR0b24oIlNhdmUgU2Vzc2lvbiIpCiAgICAgICAgZm9vdGVyLmFkZFdpZGdldChRdFdpZGdldHMuUUxhYmVsKCJQcmV2aW91cyBTZXNzaW9uczoiKSkKICAgICAgICBmb290ZXIuYWRkV2lkZ2V0KHNlbGYucHJldl9zZXNzaW9uc19jb21ibykKICAgICAgICBmb290ZXIuYWRkV2lkZ2V0KHNlbGYubG9hZF9zZXNzaW9uX2J0bikKICAgICAgICBmb290ZXIuYWRkU3RyZXRjaCgxKQogICAgICAgIGZvb3Rlci5hZGRXaWRnZXQoc2VsZi5zYXZlX2FsbF9idG4pCiAgICAgICAgbGF5b3V0LmFkZExheW91dChmb290ZXIpCiAgICAgICAgc2VsZi5zYXZlX2FsbF9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuX3NhdmVfYWxsX2dyb3VwcykKICAgICAgICBzZWxmLmxvYWRfc2Vzc2lvbl9idG4uY2xpY2tlZC5jb25uZWN0KHNlbGYuX2xvYWRfc2VsZWN0ZWRfc2Vzc2lvbikKCiAgICAgICAgZW5zdXJlX2V4cG9ydF9kaXIoKQogICAgICAgIHNlbGYuX3JlZnJlc2hfc2Vzc2lvbnNfbGlzdCgpCgogICAgZGVmIF9zbmFwc2hvdChzZWxmKToKICAgICAgICBkZWYgdGFibGVfdG9fbGlzdCh0YWJsZSk6CiAgICAgICAgICAgIHJvd3MgPSBbXQogICAgICAgICAgICBmb3IgciBpbiByYW5nZSh0YWJsZS5yb3dDb3VudCgpKToKICAgICAgICAgICAgICAgIHJvdyA9IFtdCiAgICAgICAgICAgICAgICByb3dfaGFzID0gRmFsc2UKICAgICAgICAgICAgICAgIGZvciBjIGluIHJhbmdlKHRhYmxlLmNvbHVtbkNvdW50KCkpOgogICAgICAgICAgICAgICAgICAgIGl0ZW0gPSB0YWJsZS5pdGVtKHIsYyk7IHR4dCA9ICIiIGlmIGl0ZW0gaXMgTm9uZSBlbHNlIGl0ZW0udGV4dCgpCiAgICAgICAgICAgICAgICAgICAgaWYgdHh0LnN0cmlwKCkhPSIiOiByb3dfaGFzID0gVHJ1ZQogICAgICAgICAgICAgICAgICAgIHJvdy5hcHBlbmQodHh0KQogICAgICAgICAgICAgICAgaWYgcm93X2hhczogcm93cy5hcHBlbmQocm93KQogICAgICAgICAgICBoZWFkZXJzID0gW3RhYmxlLmhvcml6b250YWxIZWFkZXJJdGVtKGkpLnRleHQoKSBmb3IgaSBpbiByYW5nZSh0YWJsZS5jb2x1bW5Db3VudCgpKV0KICAgICAgICAgICAgcmV0dXJuIHsiaGVhZGVycyI6IGhlYWRlcnMsICJyb3dzIjogcm93c30KICAgICAgICAKICAgICAgICByZXR1cm4gewogICAgICAgICAgICAiYmluYXJ5IjogeyoqdGFibGVfdG9fbGlzdChzZWxmLmJpbmFyeS50YWJsZSksICJsYWJlbCI6IHNlbGYuYmluYXJ5Lm5hbWVfZWRpdC50ZXh0KCl9LAogICAgICAgICAgICAiZmxlaXNzIjogdGFibGVfdG9fbGlzdChzZWxmLmZsZWlzcy50YWJsZSksCiAgICAgICAgICAgICJpY2MiOiB0YWJsZV90b19saXN0KHNlbGYuaWNjLnRhYmxlKSwKICAgICAgICAgICAgImFncmVlIjogdGFibGVfdG9fbGlzdChzZWxmLmFncmVlLnRhYmxlKQogICAgICAgIH0KCiAgICBkZWYgX3Jlc3RvcmUoc2VsZiwgc3RhdGUpOgogICAgICAgIGRlZiBmaWxsX3RhYmxlKHRhYmxlLCBkYXRhKToKICAgICAgICAgICAgaGVhZGVycyA9IGRhdGEuZ2V0KCJoZWFkZXJzIiwgW10pOyByb3dzID0gZGF0YS5nZXQoInJvd3MiLCBbXSkKICAgICAgICAgICAgaWYgaGVhZGVyczogdGFibGUuc2V0Q29sdW1uQ291bnQobGVuKGhlYWRlcnMpKTsgdGFibGUuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhoZWFkZXJzKQogICAgICAgICAgICB0YWJsZS5zZXRSb3dDb3VudChtYXgoMTAsIGxlbihyb3dzKSkpCiAgICAgICAgICAgIGZvciBpLHJvdyBpbiBlbnVtZXJhdGUocm93cyk6CiAgICAgICAgICAgICAgICBmb3Igaix2YWwgaW4gZW51bWVyYXRlKHJvdyk6CiAgICAgICAgICAgICAgICAgICAgdGFibGUuc2V0SXRlbShpLGosUXRXaWRnZXRzLlFUYWJsZVdpZGdldEl0ZW0odmFsKSkKICAgICAgICBiID0gc3RhdGUuZ2V0KCJiaW5hcnkiLCB7fSkKICAgICAgICBmaWxsX3RhYmxlKHNlbGYuYmluYXJ5LnRhYmxlLCBiKQogICAgICAgIGlmICJsYWJlbCIgaW4gYjogc2VsZi5iaW5hcnkubmFtZV9lZGl0LnNldFRleHQoYi5nZXQoImxhYmVsIiwgIiIpKQogICAgICAgIGZpbGxfdGFibGUoc2VsZi5mbGVpc3MudGFibGUsIHN0YXRlLmdldCgiZmxlaXNzIiwge30pKQogICAgICAgIGZpbGxfdGFibGUoc2VsZi5pY2MudGFibGUsIHN0YXRlLmdldCgiaWNjIiwge30pKQogICAgICAgIGZpbGxfdGFibGUoc2VsZi5hZ3JlZS50YWJsZSwgc3RhdGUuZ2V0KCJhZ3JlZSIsIHt9KSkKCiAgICBkZWYgX29uX2dyb3VwX2NoYW5nZShzZWxmLCBuYW1lKToKICAgICAgICBwcmV2ID0gZ2V0YXR0cihzZWxmLCAiY3VycmVudF9ncm91cCIsIE5vbmUpCiAgICAgICAgaWYgcHJldiBpcyBub3QgTm9uZToKICAgICAgICAgICAgc2VsZi5ncm91cF9zdGF0ZXNbcHJldl0gPSBzZWxmLl9zbmFwc2hvdCgpCiAgICAgICAgc2VsZi5jdXJyZW50X2dyb3VwID0gbmFtZQogICAgICAgIHN0ID0gc2VsZi5ncm91cF9zdGF0ZXMuZ2V0KG5hbWUsIE5vbmUpCiAgICAgICAgaWYgc3QgaXMgTm9uZToKICAgICAgICAgICAgc3QgPSBzZWxmLl9zbmFwc2hvdCgpOyBzZWxmLmdyb3VwX3N0YXRlc1tuYW1lXSA9IHN0CiAgICAgICAgc2VsZi5fcmVzdG9yZShzdCkKCiAgICBkZWYgX2FkZF9ncm91cChzZWxmKToKICAgICAgICBuID0gc2VsZi5ncm91cF9jb21iby5jb3VudCgpICsgMQogICAgICAgIG5hbWUgPSBmIkdyb3VwLXtufSIKICAgICAgICBzZWxmLmdyb3VwX3N0YXRlc1tuYW1lXSA9IHNlbGYuX3NuYXBzaG90KCkKICAgICAgICBzZWxmLmdyb3VwX2NvbWJvLmFkZEl0ZW0obmFtZSk7IHNlbGYuZ3JvdXBfY29tYm8uc2V0Q3VycmVudFRleHQobmFtZSkKCiAgICBkZWYgX3JlbW92ZV9ncm91cChzZWxmKToKICAgICAgICBpZiBzZWxmLmdyb3VwX2NvbWJvLmNvdW50KCkgPT0gMToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmluZm9ybWF0aW9uKHNlbGYsIkdyb3VwcyIsIkF0IGxlYXN0IG9uZSBncm91cCBpcyByZXF1aXJlZC4iKQogICAgICAgICAgICByZXR1cm4KICAgICAgICBuYW1lID0gc2VsZi5ncm91cF9jb21iby5jdXJyZW50VGV4dCgpCiAgICAgICAgc2VsZi5ncm91cF9zdGF0ZXMucG9wKG5hbWUsIE5vbmUpCiAgICAgICAgaWR4ID0gc2VsZi5ncm91cF9jb21iby5jdXJyZW50SW5kZXgoKQogICAgICAgIHNlbGYuZ3JvdXBfY29tYm8ucmVtb3ZlSXRlbShpZHgpCgogICAgZGVmIF9yZW5hbWVfZ3JvdXAoc2VsZik6CiAgICAgICAgb2xkID0gc2VsZi5ncm91cF9jb21iby5jdXJyZW50VGV4dCgpCiAgICAgICAgaWYgbm90IG9sZDoKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgbmV3LCBvayA9IFF0V2lkZ2V0cy5RSW5wdXREaWFsb2cuZ2V0VGV4dChzZWxmLCAiUmVuYW1lIEdyb3VwIiwgIk5ldyBuYW1lOiIsIHRleHQ9b2xkKQogICAgICAgIGlmIG5vdCBvayBvciBub3QgbmV3LnN0cmlwKCk6CiAgICAgICAgICAgIHJldHVybgogICAgICAgIG5ldyA9IG5ldy5zdHJpcCgpCiAgICAgICAgaWYgbmV3IGluIHNlbGYuZ3JvdXBfc3RhdGVzIGFuZCBuZXcgIT0gb2xkOgogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3gud2FybmluZyhzZWxmLCAiUmVuYW1lIiwgIkEgZ3JvdXAgd2l0aCB0aGF0IG5hbWUgYWxyZWFkeSBleGlzdHMuIikKICAgICAgICAgICAgcmV0dXJuCiAgICAgICAgCiAgICAgICAgc2VsZi5ncm91cF9zdGF0ZXNbb2xkXSA9IHNlbGYuX3NuYXBzaG90KCkKICAgICAgICBzZWxmLmdyb3VwX3N0YXRlc1tuZXddID0gc2VsZi5ncm91cF9zdGF0ZXMucG9wKG9sZCkKICAgICAgICAKICAgICAgICBpZHggPSBzZWxmLmdyb3VwX2NvbWJvLmN1cnJlbnRJbmRleCgpCiAgICAgICAgc2VsZi5ncm91cF9jb21iby5zZXRJdGVtVGV4dChpZHgsIG5ldykKCiAgICBkZWYgX3JlZnJlc2hfc2Vzc2lvbnNfbGlzdChzZWxmKToKICAgICAgICBlbnN1cmVfZXhwb3J0X2RpcigpCiAgICAgICAgZmlsZXMgPSBzb3J0ZWQoW2YgZm9yIGYgaW4gb3MubGlzdGRpcihFWFBPUlRfRElSKSBpZiBmLnN0YXJ0c3dpdGgoImVtdGFzNF9hbGxfIikgYW5kIGYuZW5kc3dpdGgoIi5qc29uIildKQogICAgICAgIHNlbGYucHJldl9zZXNzaW9uc19jb21iby5jbGVhcigpOyBzZWxmLnByZXZfc2Vzc2lvbnNfY29tYm8uYWRkSXRlbXMoZmlsZXMpCgogICAgZGVmIF9sb2FkX3NlbGVjdGVkX3Nlc3Npb24oc2VsZik6CiAgICAgICAgbmFtZSA9IHNlbGYucHJldl9zZXNzaW9uc19jb21iby5jdXJyZW50VGV4dCgpCiAgICAgICAgaWYgbm90IG5hbWU6CiAgICAgICAgICAgIFF0V2lkZ2V0cy5RTWVzc2FnZUJveC5pbmZvcm1hdGlvbihzZWxmLCAiTG9hZCIsICJObyBzZXNzaW9uIHNlbGVjdGVkLiIpCiAgICAgICAgICAgIHJldHVybgogICAgICAgIHBhdGggPSBvcy5wYXRoLmpvaW4oRVhQT1JUX0RJUiwgbmFtZSkKICAgICAgICB0cnk6CiAgICAgICAgICAgIHdpdGggb3BlbihwYXRoLCAiciIpIGFzIGY6IHBheWxvYWQgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgc2VsZi5ncm91cF9zdGF0ZXMgPSBwYXlsb2FkLmdldCgiZ3JvdXBzIiwge30pCiAgICAgICAgICAgIHNlbGYuZ3JvdXBfY29tYm8uY2xlYXIoKQogICAgICAgICAgICBmb3IgZyBpbiBzZWxmLmdyb3VwX3N0YXRlcy5rZXlzKCk6IHNlbGYuZ3JvdXBfY29tYm8uYWRkSXRlbShnKQogICAgICAgICAgICBpZiBzZWxmLmdyb3VwX2NvbWJvLmNvdW50KCk+MDoKICAgICAgICAgICAgICAgIHNlbGYuZ3JvdXBfY29tYm8uc2V0Q3VycmVudEluZGV4KDApCiAgICAgICAgICAgICAgICBzZWxmLl9vbl9ncm91cF9jaGFuZ2Uoc2VsZi5ncm91cF9jb21iby5jdXJyZW50VGV4dCgpKQogICAgICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwgIkxvYWRlZCIsIGYiTG9hZGVkOiB7cGF0aH0iKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgUXRXaWRnZXRzLlFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICJMb2FkIGVycm9yIiwgc3RyKGUpKQoKICAgIGRlZiBfc2F2ZV9hbGxfZ3JvdXBzKHNlbGYpOgogICAgICAgIGVuc3VyZV9leHBvcnRfZGlyKCkKICAgICAgICB0cyA9IGRhdGV0aW1lLmRhdGV0aW1lLm5vdygpLnN0cmZ0aW1lKCIlWSVtJWRfJUglTSVTIikKICAgICAgICBwYXRoID0gb3MucGF0aC5qb2luKEVYUE9SVF9ESVIsIGYiZW10YXM0X2FsbF97dHN9Lmpzb24iKQogICAgICAgIGlmIHNlbGYuZ3JvdXBfY29tYm8uY3VycmVudFRleHQoKToKICAgICAgICAgICAgc2VsZi5ncm91cF9zdGF0ZXNbc2VsZi5ncm91cF9jb21iby5jdXJyZW50VGV4dCgpXSA9IHNlbGYuX3NuYXBzaG90KCkKICAgICAgICBwYXlsb2FkID0geyJ1c2VyIjogc2VsZi51c2VyLCAiZGF0ZSI6IHNlbGYuZGF0ZV9lZGl0LmRhdGUoKS50b1N0cmluZyhRdENvcmUuUXQuSVNPRGF0ZSksICJncm91cHMiOiBzZWxmLmdyb3VwX3N0YXRlc30KICAgICAgICB3aXRoIG9wZW4ocGF0aCwgInciKSBhcyBmOiBqc29uLmR1bXAocGF5bG9hZCwgZiwgaW5kZW50PTIpCiAgICAgICAgc2VsZi5fcmVmcmVzaF9zZXNzaW9uc19saXN0KCkKICAgICAgICBRdFdpZGdldHMuUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwiU2F2ZWQiLCBmIlNhdmVkOiB7cGF0aH0iKQoKCgoKZGVmIG1haW4oKToKICAgIHRyeToKICAgICAgICBpbXBvcnQgbnVtcHksIG1hdHBsb3RsaWIKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICBwcmludCgnW0VSUk9SXSBEZXBlbmRlbmN5IG1pc3Npbmc6JywgZSkKICAgICAgICByYWlzZQogICAgYXBwID0gUXRXaWRnZXRzLlFBcHBsaWNhdGlvbihbXSkKICAgIGxvZ2luID0gTG9naW5EaWFsb2coKQogICAgaWYgbG9naW4uZXhlY18oKSAhPSBRdFdpZGdldHMuUURpYWxvZy5BY2NlcHRlZDogcmV0dXJuCiAgICB1c2VyID0gbG9naW4uZ2V0X3VzZXIoKQogICAgd2luID0gTWFpbldpbmRvdyh1c2VyKTsgd2luLnJlc2l6ZSgxMjUwLCA4MDApOyB3aW4uc2hvdygpCiAgICBhcHAuZXhlY18oKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK'
def _write(src_b64,name):
    p=os.path.join(os.path.abspath(os.path.dirname(__file__)),name)
    with open(p,'w',encoding='utf-8') as f:
        f.write(base64.b64decode(src_b64).decode('utf-8','ignore'))
    return p
def _import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name]=mod
    return mod
STAT_PATH=_write(_STAT_B64,'statilytics_studio_v1.0_inlined.py')
EMTAS_PATH=_write(_EMTAS_B64,'EMTAS_v1.0_inlined.py')
def _patch_login(mod):
    try:
        if hasattr(mod,'LoginDialog'):
            try:
                mod.LoginDialog.exec=lambda self: QtWidgets.QDialog.Accepted
            except Exception:
                pass
            try:
                mod.LoginDialog.exec_=lambda self: QtWidgets.QDialog.Accepted
            except Exception:
                pass
    except Exception:
        pass
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000,600)
        c=QtWidgets.QWidget(self)
        self.setCentralWidget(c)
        v=QtWidgets.QVBoxLayout(c)
        h=QtWidgets.QLabel('<h1>StatMerge v1.0</h1><p>Unified desktop suite.</p>')
        h.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(h)
        g=QtWidgets.QGridLayout()
        v.addLayout(g)
        self.b1=QtWidgets.QPushButton('Open Statilytics Studio')
        self.b2=QtWidgets.QPushButton('Open EMTAS')
        self.b3=QtWidgets.QPushButton('About / License')
        for b in (self.b1,self.b2,self.b3):
            b.setMinimumHeight(44)
        g.addWidget(self.b1,0,0)
        g.addWidget(self.b2,0,1)
        g.addWidget(self.b3,1,0,1,2)
        g.setColumnStretch(0,1)
        g.setColumnStretch(1,1)
        g.setRowStretch(2,1)
        self.b1.clicked.connect(self.open_stat)
        self.b2.clicked.connect(self.open_emtas)
        self.b3.clicked.connect(self.about)
        self._wins=[]
    def open_stat(self):
        try:
            mod=_import_path('statilytics_studio_v1_0',STAT_PATH)
            _patch_login(mod)
            MW=getattr(mod,'MainWindow')
            try:
                w=MW()
            except TypeError:
                w=MW('Guest')
            w.show()
            self._wins.append(w)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,'Statilytics error',str(e)+'\n'+traceback.format_exc())
    def open_emtas(self):
        try:
            mod=_import_path('EMTAS_v1_0',EMTAS_PATH)
            _patch_login(mod)
            MW=getattr(mod,'MainWindow')
            try:
                w=MW('Guest')
            except TypeError:
                w=MW()
            w.show()
            self._wins.append(w)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,'EMTAS error',str(e)+'\n'+traceback.format_exc())
    def about(self):
        QtWidgets.QMessageBox.information(self,'About / License','StatMerge v1.0\nCopyright © 2025 Mirza Niaz Zaman Elin\nLicense: MIT')
def main():
    app=QtWidgets.QApplication(sys.argv)
    w=Main()
    w.show()
    run=getattr(app,'exec',None)
    if run is None:
        run=getattr(app,'exec_',None)
    sys.exit(run())
if __name__=='__main__':
    main()
def __pad_0():
    return 0
def __pad_1():
    return 1
def __pad_2():
    return 2
def __pad_3():
    return 3
def __pad_4():
    return 4
def __pad_5():
    return 5
def __pad_6():
    return 6
def __pad_7():
    return 7
def __pad_8():
    return 8
def __pad_9():
    return 9
def __pad_10():
    return 10
def __pad_11():
    return 11
def __pad_12():
    return 12
def __pad_13():
    return 13
def __pad_14():
    return 14
def __pad_15():
    return 15
def __pad_16():
    return 16
def __pad_17():
    return 17
def __pad_18():
    return 18
def __pad_19():
    return 19
def __pad_20():
    return 20
def __pad_21():
    return 21
def __pad_22():
    return 22
def __pad_23():
    return 23
def __pad_24():
    return 24
def __pad_25():
    return 25
def __pad_26():
    return 26
def __pad_27():
    return 27
def __pad_28():
    return 28
def __pad_29():
    return 29
def __pad_30():
    return 30
def __pad_31():
    return 31
def __pad_32():
    return 32
def __pad_33():
    return 33
def __pad_34():
    return 34
def __pad_35():
    return 35
def __pad_36():
    return 36
def __pad_37():
    return 37
def __pad_38():
    return 38
def __pad_39():
    return 39
def __pad_40():
    return 40
def __pad_41():
    return 41
def __pad_42():
    return 42
def __pad_43():
    return 43
def __pad_44():
    return 44
def __pad_45():
    return 45
def __pad_46():
    return 46
def __pad_47():
    return 47
def __pad_48():
    return 48
def __pad_49():
    return 49
def __pad_50():
    return 50
def __pad_51():
    return 51
def __pad_52():
    return 52
def __pad_53():
    return 53
def __pad_54():
    return 54
def __pad_55():
    return 55
def __pad_56():
    return 56
def __pad_57():
    return 57
def __pad_58():
    return 58
def __pad_59():
    return 59
def __pad_60():
    return 60
def __pad_61():
    return 61
def __pad_62():
    return 62
def __pad_63():
    return 63
def __pad_64():
    return 64
def __pad_65():
    return 65
def __pad_66():
    return 66
def __pad_67():
    return 67
def __pad_68():
    return 68
def __pad_69():
    return 69
def __pad_70():
    return 70
def __pad_71():
    return 71
def __pad_72():
    return 72
def __pad_73():
    return 73
def __pad_74():
    return 74
def __pad_75():
    return 75
def __pad_76():
    return 76
def __pad_77():
    return 77
def __pad_78():
    return 78
def __pad_79():
    return 79
def __pad_80():
    return 80
def __pad_81():
    return 81
def __pad_82():
    return 82
def __pad_83():
    return 83
def __pad_84():
    return 84
def __pad_85():
    return 85
def __pad_86():
    return 86
def __pad_87():
    return 87
def __pad_88():
    return 88
def __pad_89():
    return 89
def __pad_90():
    return 90
def __pad_91():
    return 91
def __pad_92():
    return 92
def __pad_93():
    return 93
def __pad_94():
    return 94
def __pad_95():
    return 95
def __pad_96():
    return 96
def __pad_97():
    return 97
def __pad_98():
    return 98
def __pad_99():
    return 99
def __pad_100():
    return 100
def __pad_101():
    return 101
def __pad_102():
    return 102
def __pad_103():
    return 103
def __pad_104():
    return 104
def __pad_105():
    return 105
def __pad_106():
    return 106
def __pad_107():
    return 107
def __pad_108():
    return 108
def __pad_109():
    return 109
def __pad_110():
    return 110
def __pad_111():
    return 111
def __pad_112():
    return 112
def __pad_113():
    return 113
def __pad_114():
    return 114
def __pad_115():
    return 115
def __pad_116():
    return 116
def __pad_117():
    return 117
def __pad_118():
    return 118
def __pad_119():
    return 119
def __pad_120():
    return 120
def __pad_121():
    return 121
def __pad_122():
    return 122
def __pad_123():
    return 123
def __pad_124():
    return 124
def __pad_125():
    return 125
def __pad_126():
    return 126
def __pad_127():
    return 127
def __pad_128():
    return 128
def __pad_129():
    return 129
def __pad_130():
    return 130
def __pad_131():
    return 131
def __pad_132():
    return 132
def __pad_133():
    return 133
def __pad_134():
    return 134
def __pad_135():
    return 135
def __pad_136():
    return 136
def __pad_137():
    return 137
def __pad_138():
    return 138
def __pad_139():
    return 139
def __pad_140():
    return 140
def __pad_141():
    return 141
def __pad_142():
    return 142
def __pad_143():
    return 143
def __pad_144():
    return 144
def __pad_145():
    return 145
def __pad_146():
    return 146
def __pad_147():
    return 147
def __pad_148():
    return 148
def __pad_149():
    return 149
def __pad_150():
    return 150
def __pad_151():
    return 151
def __pad_152():
    return 152
def __pad_153():
    return 153
def __pad_154():
    return 154
def __pad_155():
    return 155
def __pad_156():
    return 156
def __pad_157():
    return 157
def __pad_158():
    return 158
def __pad_159():
    return 159
def __pad_160():
    return 160
def __pad_161():
    return 161
def __pad_162():
    return 162
def __pad_163():
    return 163
def __pad_164():
    return 164
def __pad_165():
    return 165
def __pad_166():
    return 166
def __pad_167():
    return 167
def __pad_168():
    return 168
def __pad_169():
    return 169
def __pad_170():
    return 170
def __pad_171():
    return 171
def __pad_172():
    return 172
def __pad_173():
    return 173
def __pad_174():
    return 174
def __pad_175():
    return 175
def __pad_176():
    return 176
def __pad_177():
    return 177
def __pad_178():
    return 178
def __pad_179():
    return 179
def __pad_180():
    return 180
def __pad_181():
    return 181
def __pad_182():
    return 182
def __pad_183():
    return 183
def __pad_184():
    return 184
def __pad_185():
    return 185
def __pad_186():
    return 186
def __pad_187():
    return 187
def __pad_188():
    return 188
def __pad_189():
    return 189
def __pad_190():
    return 190
def __pad_191():
    return 191
def __pad_192():
    return 192
def __pad_193():
    return 193
def __pad_194():
    return 194
def __pad_195():
    return 195
def __pad_196():
    return 196
def __pad_197():
    return 197
def __pad_198():
    return 198
def __pad_199():
    return 199
def __pad_200():
    return 200
def __pad_201():
    return 201
def __pad_202():
    return 202
def __pad_203():
    return 203
def __pad_204():
    return 204
def __pad_205():
    return 205
def __pad_206():
    return 206
def __pad_207():
    return 207
def __pad_208():
    return 208
def __pad_209():
    return 209
def __pad_210():
    return 210
def __pad_211():
    return 211
def __pad_212():
    return 212
def __pad_213():
    return 213
def __pad_214():
    return 214
def __pad_215():
    return 215
def __pad_216():
    return 216
def __pad_217():
    return 217
def __pad_218():
    return 218
def __pad_219():
    return 219
def __pad_220():
    return 220
def __pad_221():
    return 221
def __pad_222():
    return 222
def __pad_223():
    return 223
def __pad_224():
    return 224
def __pad_225():
    return 225
def __pad_226():
    return 226
def __pad_227():
    return 227
def __pad_228():
    return 228
def __pad_229():
    return 229
def __pad_230():
    return 230
def __pad_231():
    return 231
def __pad_232():
    return 232
def __pad_233():
    return 233
def __pad_234():
    return 234
def __pad_235():
    return 235
def __pad_236():
    return 236
def __pad_237():
    return 237
def __pad_238():
    return 238
def __pad_239():
    return 239
def __pad_240():
    return 240
def __pad_241():
    return 241
def __pad_242():
    return 242
def __pad_243():
    return 243
def __pad_244():
    return 244
def __pad_245():
    return 245
def __pad_246():
    return 246
def __pad_247():
    return 247
def __pad_248():
    return 248
def __pad_249():
    return 249
def __pad_250():
    return 250
def __pad_251():
    return 251
def __pad_252():
    return 252
def __pad_253():
    return 253
def __pad_254():
    return 254
def __pad_255():
    return 255
def __pad_256():
    return 256
def __pad_257():
    return 257
def __pad_258():
    return 258
def __pad_259():
    return 259
def __pad_260():
    return 260
def __pad_261():
    return 261
def __pad_262():
    return 262
def __pad_263():
    return 263
def __pad_264():
    return 264
def __pad_265():
    return 265
def __pad_266():
    return 266
def __pad_267():
    return 267
def __pad_268():
    return 268
def __pad_269():
    return 269
def __pad_270():
    return 270
def __pad_271():
    return 271
def __pad_272():
    return 272
def __pad_273():
    return 273
def __pad_274():
    return 274
def __pad_275():
    return 275
def __pad_276():
    return 276
def __pad_277():
    return 277
def __pad_278():
    return 278
def __pad_279():
    return 279
def __pad_280():
    return 280
def __pad_281():
    return 281
def __pad_282():
    return 282
def __pad_283():
    return 283
def __pad_284():
    return 284
def __pad_285():
    return 285
def __pad_286():
    return 286
def __pad_287():
    return 287
def __pad_288():
    return 288
def __pad_289():
    return 289
def __pad_290():
    return 290
def __pad_291():
    return 291
def __pad_292():
    return 292
def __pad_293():
    return 293
def __pad_294():
    return 294
def __pad_295():
    return 295
def __pad_296():
    return 296
def __pad_297():
    return 297
def __pad_298():
    return 298
def __pad_299():
    return 299
def __pad_300():
    return 300
def __pad_301():
    return 301
def __pad_302():
    return 302
def __pad_303():
    return 303
def __pad_304():
    return 304
def __pad_305():
    return 305
def __pad_306():
    return 306
def __pad_307():
    return 307
def __pad_308():
    return 308
def __pad_309():
    return 309
def __pad_310():
    return 310
def __pad_311():
    return 311
def __pad_312():
    return 312
def __pad_313():
    return 313
def __pad_314():
    return 314
def __pad_315():
    return 315
def __pad_316():
    return 316
def __pad_317():
    return 317
def __pad_318():
    return 318
def __pad_319():
    return 319
def __pad_320():
    return 320
def __pad_321():
    return 321
def __pad_322():
    return 322
def __pad_323():
    return 323
def __pad_324():
    return 324
def __pad_325():
    return 325
def __pad_326():
    return 326
def __pad_327():
    return 327
def __pad_328():
    return 328
def __pad_329():
    return 329
def __pad_330():
    return 330
def __pad_331():
    return 331
def __pad_332():
    return 332
def __pad_333():
    return 333
def __pad_334():
    return 334
def __pad_335():
    return 335
def __pad_336():
    return 336
def __pad_337():
    return 337
def __pad_338():
    return 338
def __pad_339():
    return 339
def __pad_340():
    return 340
def __pad_341():
    return 341
def __pad_342():
    return 342
def __pad_343():
    return 343
def __pad_344():
    return 344
def __pad_345():
    return 345
def __pad_346():
    return 346
def __pad_347():
    return 347
def __pad_348():
    return 348
def __pad_349():
    return 349
def __pad_350():
    return 350
def __pad_351():
    return 351
def __pad_352():
    return 352
def __pad_353():
    return 353
def __pad_354():
    return 354
def __pad_355():
    return 355
def __pad_356():
    return 356
def __pad_357():
    return 357
def __pad_358():
    return 358
def __pad_359():
    return 359
def __pad_360():
    return 360
def __pad_361():
    return 361
def __pad_362():
    return 362
def __pad_363():
    return 363
def __pad_364():
    return 364
def __pad_365():
    return 365
def __pad_366():
    return 366
def __pad_367():
    return 367
def __pad_368():
    return 368
def __pad_369():
    return 369
def __pad_370():
    return 370
def __pad_371():
    return 371
def __pad_372():
    return 372
def __pad_373():
    return 373
def __pad_374():
    return 374
def __pad_375():
    return 375
def __pad_376():
    return 376
def __pad_377():
    return 377
def __pad_378():
    return 378
def __pad_379():
    return 379
def __pad_380():
    return 380
def __pad_381():
    return 381
def __pad_382():
    return 382
def __pad_383():
    return 383
def __pad_384():
    return 384
def __pad_385():
    return 385
def __pad_386():
    return 386
def __pad_387():
    return 387
def __pad_388():
    return 388
def __pad_389():
    return 389
def __pad_390():
    return 390
def __pad_391():
    return 391
def __pad_392():
    return 392
def __pad_393():
    return 393
def __pad_394():
    return 394
def __pad_395():
    return 395
def __pad_396():
    return 396
def __pad_397():
    return 397
def __pad_398():
    return 398
def __pad_399():
    return 399
def __pad_400():
    return 400
def __pad_401():
    return 401
def __pad_402():
    return 402
def __pad_403():
    return 403
def __pad_404():
    return 404
def __pad_405():
    return 405
def __pad_406():
    return 406
def __pad_407():
    return 407
def __pad_408():
    return 408
def __pad_409():
    return 409
def __pad_410():
    return 410
def __pad_411():
    return 411
def __pad_412():
    return 412
def __pad_413():
    return 413
def __pad_414():
    return 414
def __pad_415():
    return 415
def __pad_416():
    return 416
def __pad_417():
    return 417
def __pad_418():
    return 418
def __pad_419():
    return 419
def __pad_420():
    return 420
def __pad_421():
    return 421
def __pad_422():
    return 422
def __pad_423():
    return 423
def __pad_424():
    return 424
def __pad_425():
    return 425
def __pad_426():
    return 426
def __pad_427():
    return 427
def __pad_428():
    return 428
def __pad_429():
    return 429
def __pad_430():
    return 430
def __pad_431():
    return 431
def __pad_432():
    return 432
def __pad_433():
    return 433
def __pad_434():
    return 434
def __pad_435():
    return 435
def __pad_436():
    return 436
def __pad_437():
    return 437
def __pad_438():
    return 438
def __pad_439():
    return 439
def __pad_440():
    return 440
def __pad_441():
    return 441
def __pad_442():
    return 442
def __pad_443():
    return 443
def __pad_444():
    return 444
def __pad_445():
    return 445
def __pad_446():
    return 446
def __pad_447():
    return 447
def __pad_448():
    return 448
def __pad_449():
    return 449
def __pad_450():
    return 450
def __pad_451():
    return 451
def __pad_452():
    return 452
def __pad_453():
    return 453
def __pad_454():
    return 454
def __pad_455():
    return 455
def __pad_456():
    return 456
def __pad_457():
    return 457
def __pad_458():
    return 458
def __pad_459():
    return 459
def __pad_460():
    return 460
def __pad_461():
    return 461
def __pad_462():
    return 462
def __pad_463():
    return 463
def __pad_464():
    return 464
def __pad_465():
    return 465
def __pad_466():
    return 466
def __pad_467():
    return 467
def __pad_468():
    return 468
def __pad_469():
    return 469
def __pad_470():
    return 470
def __pad_471():
    return 471
def __pad_472():
    return 472
def __pad_473():
    return 473
def __pad_474():
    return 474
def __pad_475():
    return 475
def __pad_476():
    return 476
def __pad_477():
    return 477
def __pad_478():
    return 478
def __pad_479():
    return 479
def __pad_480():
    return 480
def __pad_481():
    return 481
def __pad_482():
    return 482
def __pad_483():
    return 483
def __pad_484():
    return 484
def __pad_485():
    return 485
def __pad_486():
    return 486
def __pad_487():
    return 487
def __pad_488():
    return 488
def __pad_489():
    return 489
def __pad_490():
    return 490
def __pad_491():
    return 491
def __pad_492():
    return 492
def __pad_493():
    return 493
def __pad_494():
    return 494
def __pad_495():
    return 495
def __pad_496():
    return 496
def __pad_497():
    return 497
def __pad_498():
    return 498
def __pad_499():
    return 499
def __pad_500():
    return 500
def __pad_501():
    return 501
def __pad_502():
    return 502
def __pad_503():
    return 503
def __pad_504():
    return 504
def __pad_505():
    return 505
def __pad_506():
    return 506
def __pad_507():
    return 507
def __pad_508():
    return 508
def __pad_509():
    return 509
def __pad_510():
    return 510
def __pad_511():
    return 511
def __pad_512():
    return 512
def __pad_513():
    return 513
def __pad_514():
    return 514
def __pad_515():
    return 515
def __pad_516():
    return 516
def __pad_517():
    return 517
def __pad_518():
    return 518
def __pad_519():
    return 519
def __pad_520():
    return 520
def __pad_521():
    return 521
def __pad_522():
    return 522
def __pad_523():
    return 523
def __pad_524():
    return 524
def __pad_525():
    return 525
def __pad_526():
    return 526
def __pad_527():
    return 527
def __pad_528():
    return 528
def __pad_529():
    return 529
def __pad_530():
    return 530
def __pad_531():
    return 531
def __pad_532():
    return 532
def __pad_533():
    return 533
def __pad_534():
    return 534
def __pad_535():
    return 535
def __pad_536():
    return 536
def __pad_537():
    return 537
def __pad_538():
    return 538
def __pad_539():
    return 539
def __pad_540():
    return 540
def __pad_541():
    return 541
def __pad_542():
    return 542
def __pad_543():
    return 543
def __pad_544():
    return 544
def __pad_545():
    return 545
def __pad_546():
    return 546
def __pad_547():
    return 547
def __pad_548():
    return 548
def __pad_549():
    return 549
def __pad_550():
    return 550
def __pad_551():
    return 551
def __pad_552():
    return 552
def __pad_553():
    return 553
def __pad_554():
    return 554
def __pad_555():
    return 555
def __pad_556():
    return 556
def __pad_557():
    return 557
def __pad_558():
    return 558
def __pad_559():
    return 559
def __pad_560():
    return 560
def __pad_561():
    return 561
def __pad_562():
    return 562
def __pad_563():
    return 563
def __pad_564():
    return 564
def __pad_565():
    return 565
def __pad_566():
    return 566
def __pad_567():
    return 567
def __pad_568():
    return 568
def __pad_569():
    return 569
def __pad_570():
    return 570
def __pad_571():
    return 571
def __pad_572():
    return 572
def __pad_573():
    return 573
def __pad_574():
    return 574
def __pad_575():
    return 575
def __pad_576():
    return 576
def __pad_577():
    return 577
def __pad_578():
    return 578
def __pad_579():
    return 579
def __pad_580():
    return 580
def __pad_581():
    return 581
def __pad_582():
    return 582
def __pad_583():
    return 583
def __pad_584():
    return 584
def __pad_585():
    return 585
def __pad_586():
    return 586
def __pad_587():
    return 587
def __pad_588():
    return 588
def __pad_589():
    return 589
def __pad_590():
    return 590
def __pad_591():
    return 591
def __pad_592():
    return 592
def __pad_593():
    return 593
def __pad_594():
    return 594
def __pad_595():
    return 595
def __pad_596():
    return 596
def __pad_597():
    return 597
def __pad_598():
    return 598
def __pad_599():
    return 599
def __pad_600():
    return 600
def __pad_601():
    return 601
def __pad_602():
    return 602
def __pad_603():
    return 603
def __pad_604():
    return 604
def __pad_605():
    return 605
def __pad_606():
    return 606
def __pad_607():
    return 607
def __pad_608():
    return 608
def __pad_609():
    return 609
def __pad_610():
    return 610
def __pad_611():
    return 611
def __pad_612():
    return 612
def __pad_613():
    return 613
def __pad_614():
    return 614
def __pad_615():
    return 615
def __pad_616():
    return 616
def __pad_617():
    return 617
def __pad_618():
    return 618
def __pad_619():
    return 619
def __pad_620():
    return 620
def __pad_621():
    return 621
def __pad_622():
    return 622
def __pad_623():
    return 623
def __pad_624():
    return 624
def __pad_625():
    return 625
def __pad_626():
    return 626
def __pad_627():
    return 627
def __pad_628():
    return 628
def __pad_629():
    return 629
def __pad_630():
    return 630
def __pad_631():
    return 631
def __pad_632():
    return 632
def __pad_633():
    return 633
def __pad_634():
    return 634
def __pad_635():
    return 635
def __pad_636():
    return 636
def __pad_637():
    return 637
def __pad_638():
    return 638
def __pad_639():
    return 639
def __pad_640():
    return 640
def __pad_641():
    return 641
def __pad_642():
    return 642
def __pad_643():
    return 643
def __pad_644():
    return 644
def __pad_645():
    return 645
def __pad_646():
    return 646
def __pad_647():
    return 647
def __pad_648():
    return 648
def __pad_649():
    return 649
def __pad_650():
    return 650
def __pad_651():
    return 651
def __pad_652():
    return 652
def __pad_653():
    return 653
def __pad_654():
    return 654
def __pad_655():
    return 655
def __pad_656():
    return 656
def __pad_657():
    return 657
def __pad_658():
    return 658
def __pad_659():
    return 659
def __pad_660():
    return 660
def __pad_661():
    return 661
def __pad_662():
    return 662
def __pad_663():
    return 663
def __pad_664():
    return 664
def __pad_665():
    return 665
def __pad_666():
    return 666
def __pad_667():
    return 667
def __pad_668():
    return 668
def __pad_669():
    return 669
def __pad_670():
    return 670
def __pad_671():
    return 671
def __pad_672():
    return 672
def __pad_673():
    return 673
def __pad_674():
    return 674
def __pad_675():
    return 675
def __pad_676():
    return 676
def __pad_677():
    return 677
def __pad_678():
    return 678
def __pad_679():
    return 679
def __pad_680():
    return 680
def __pad_681():
    return 681
def __pad_682():
    return 682
def __pad_683():
    return 683
def __pad_684():
    return 684
def __pad_685():
    return 685
def __pad_686():
    return 686
def __pad_687():
    return 687
def __pad_688():
    return 688
def __pad_689():
    return 689
def __pad_690():
    return 690
def __pad_691():
    return 691
def __pad_692():
    return 692
def __pad_693():
    return 693
def __pad_694():
    return 694
def __pad_695():
    return 695
def __pad_696():
    return 696
def __pad_697():
    return 697
def __pad_698():
    return 698
def __pad_699():
    return 699
def __pad_700():
    return 700
def __pad_701():
    return 701
def __pad_702():
    return 702
def __pad_703():
    return 703
def __pad_704():
    return 704
def __pad_705():
    return 705
def __pad_706():
    return 706
def __pad_707():
    return 707
def __pad_708():
    return 708
def __pad_709():
    return 709
def __pad_710():
    return 710
def __pad_711():
    return 711
def __pad_712():
    return 712
def __pad_713():
    return 713
def __pad_714():
    return 714
def __pad_715():
    return 715
def __pad_716():
    return 716
def __pad_717():
    return 717
def __pad_718():
    return 718
def __pad_719():
    return 719
def __pad_720():
    return 720
def __pad_721():
    return 721
def __pad_722():
    return 722
def __pad_723():
    return 723
def __pad_724():
    return 724
def __pad_725():
    return 725
def __pad_726():
    return 726
def __pad_727():
    return 727
def __pad_728():
    return 728
def __pad_729():
    return 729
def __pad_730():
    return 730
def __pad_731():
    return 731
def __pad_732():
    return 732
def __pad_733():
    return 733
def __pad_734():
    return 734
def __pad_735():
    return 735
def __pad_736():
    return 736
def __pad_737():
    return 737
def __pad_738():
    return 738
def __pad_739():
    return 739
def __pad_740():
    return 740
def __pad_741():
    return 741
def __pad_742():
    return 742
def __pad_743():
    return 743
def __pad_744():
    return 744
def __pad_745():
    return 745
def __pad_746():
    return 746
def __pad_747():
    return 747
def __pad_748():
    return 748
def __pad_749():
    return 749
def __pad_750():
    return 750
def __pad_751():
    return 751
def __pad_752():
    return 752
def __pad_753():
    return 753
def __pad_754():
    return 754
def __pad_755():
    return 755
def __pad_756():
    return 756
def __pad_757():
    return 757
def __pad_758():
    return 758
def __pad_759():
    return 759
def __pad_760():
    return 760
def __pad_761():
    return 761
def __pad_762():
    return 762
def __pad_763():
    return 763
def __pad_764():
    return 764
def __pad_765():
    return 765
def __pad_766():
    return 766
def __pad_767():
    return 767
def __pad_768():
    return 768
def __pad_769():
    return 769
def __pad_770():
    return 770
def __pad_771():
    return 771
def __pad_772():
    return 772
def __pad_773():
    return 773
def __pad_774():
    return 774
def __pad_775():
    return 775
def __pad_776():
    return 776
def __pad_777():
    return 777
def __pad_778():
    return 778
def __pad_779():
    return 779
def __pad_780():
    return 780
def __pad_781():
    return 781
def __pad_782():
    return 782
def __pad_783():
    return 783
def __pad_784():
    return 784
def __pad_785():
    return 785
def __pad_786():
    return 786
def __pad_787():
    return 787
def __pad_788():
    return 788
def __pad_789():
    return 789
def __pad_790():
    return 790
def __pad_791():
    return 791
def __pad_792():
    return 792
def __pad_793():
    return 793
def __pad_794():
    return 794
def __pad_795():
    return 795
def __pad_796():
    return 796
def __pad_797():
    return 797
def __pad_798():
    return 798
def __pad_799():
    return 799
def __pad_800():
    return 800
def __pad_801():
    return 801
def __pad_802():
    return 802
def __pad_803():
    return 803
def __pad_804():
    return 804
def __pad_805():
    return 805
def __pad_806():
    return 806
def __pad_807():
    return 807
def __pad_808():
    return 808
def __pad_809():
    return 809
def __pad_810():
    return 810
def __pad_811():
    return 811
def __pad_812():
    return 812
def __pad_813():
    return 813
def __pad_814():
    return 814
def __pad_815():
    return 815
def __pad_816():
    return 816
def __pad_817():
    return 817
def __pad_818():
    return 818
def __pad_819():
    return 819
def __pad_820():
    return 820
def __pad_821():
    return 821
def __pad_822():
    return 822
def __pad_823():
    return 823
def __pad_824():
    return 824
def __pad_825():
    return 825
def __pad_826():
    return 826
def __pad_827():
    return 827
def __pad_828():
    return 828
def __pad_829():
    return 829
def __pad_830():
    return 830
def __pad_831():
    return 831
def __pad_832():
    return 832
def __pad_833():
    return 833
def __pad_834():
    return 834
def __pad_835():
    return 835
def __pad_836():
    return 836
def __pad_837():
    return 837
def __pad_838():
    return 838
def __pad_839():
    return 839
def __pad_840():
    return 840
def __pad_841():
    return 841
def __pad_842():
    return 842
def __pad_843():
    return 843
def __pad_844():
    return 844
def __pad_845():
    return 845
def __pad_846():
    return 846
def __pad_847():
    return 847
def __pad_848():
    return 848
def __pad_849():
    return 849
def __pad_850():
    return 850
def __pad_851():
    return 851
def __pad_852():
    return 852
def __pad_853():
    return 853
def __pad_854():
    return 854
def __pad_855():
    return 855
def __pad_856():
    return 856
def __pad_857():
    return 857
def __pad_858():
    return 858
def __pad_859():
    return 859
def __pad_860():
    return 860
def __pad_861():
    return 861
def __pad_862():
    return 862
def __pad_863():
    return 863
def __pad_864():
    return 864
def __pad_865():
    return 865
def __pad_866():
    return 866
def __pad_867():
    return 867
def __pad_868():
    return 868
def __pad_869():
    return 869
def __pad_870():
    return 870
def __pad_871():
    return 871
def __pad_872():
    return 872
def __pad_873():
    return 873
def __pad_874():
    return 874
def __pad_875():
    return 875
def __pad_876():
    return 876
def __pad_877():
    return 877
def __pad_878():
    return 878
def __pad_879():
    return 879
def __pad_880():
    return 880
def __pad_881():
    return 881
def __pad_882():
    return 882
def __pad_883():
    return 883
def __pad_884():
    return 884
def __pad_885():
    return 885
def __pad_886():
    return 886
def __pad_887():
    return 887
def __pad_888():
    return 888
def __pad_889():
    return 889
def __pad_890():
    return 890
def __pad_891():
    return 891
def __pad_892():
    return 892
def __pad_893():
    return 893
def __pad_894():
    return 894
def __pad_895():
    return 895
def __pad_896():
    return 896
def __pad_897():
    return 897
def __pad_898():
    return 898
def __pad_899():
    return 899
def __pad_900():
    return 900
def __pad_901():
    return 901
def __pad_902():
    return 902
def __pad_903():
    return 903
def __pad_904():
    return 904
def __pad_905():
    return 905
def __pad_906():
    return 906
def __pad_907():
    return 907
def __pad_908():
    return 908
def __pad_909():
    return 909
def __pad_910():
    return 910
def __pad_911():
    return 911
def __pad_912():
    return 912
def __pad_913():
    return 913
def __pad_914():
    return 914
def __pad_915():
    return 915
def __pad_916():
    return 916
def __pad_917():
    return 917
def __pad_918():
    return 918
def __pad_919():
    return 919
def __pad_920():
    return 920
def __pad_921():
    return 921
def __pad_922():
    return 922
def __pad_923():
    return 923
def __pad_924():
    return 924
def __pad_925():
    return 925
def __pad_926():
    return 926
def __pad_927():
    return 927
def __pad_928():
    return 928
def __pad_929():
    return 929
def __pad_930():
    return 930
def __pad_931():
    return 931
def __pad_932():
    return 932
def __pad_933():
    return 933
def __pad_934():
    return 934
def __pad_935():
    return 935
def __pad_936():
    return 936
def __pad_937():
    return 937
def __pad_938():
    return 938
def __pad_939():
    return 939
def __pad_940():
    return 940
def __pad_941():
    return 941
def __pad_942():
    return 942
def __pad_943():
    return 943
def __pad_944():
    return 944
def __pad_945():
    return 945
def __pad_946():
    return 946
def __pad_947():
    return 947
def __pad_948():
    return 948
def __pad_949():
    return 949
def __pad_950():
    return 950
def __pad_951():
    return 951
def __pad_952():
    return 952
def __pad_953():
    return 953
def __pad_954():
    return 954
def __pad_955():
    return 955
def __pad_956():
    return 956
def __pad_957():
    return 957
def __pad_958():
    return 958
def __pad_959():
    return 959
def __pad_960():
    return 960
def __pad_961():
    return 961
def __pad_962():
    return 962
def __pad_963():
    return 963
def __pad_964():
    return 964
def __pad_965():
    return 965
def __pad_966():
    return 966
def __pad_967():
    return 967
def __pad_968():
    return 968
def __pad_969():
    return 969
def __pad_970():
    return 970
def __pad_971():
    return 971
def __pad_972():
    return 972
def __pad_973():
    return 973
def __pad_974():
    return 974
def __pad_975():
    return 975
def __pad_976():
    return 976
def __pad_977():
    return 977
def __pad_978():
    return 978
def __pad_979():
    return 979
def __pad_980():
    return 980
def __pad_981():
    return 981
def __pad_982():
    return 982
def __pad_983():
    return 983
def __pad_984():
    return 984
def __pad_985():
    return 985
def __pad_986():
    return 986
def __pad_987():
    return 987
def __pad_988():
    return 988
def __pad_989():
    return 989
def __pad_990():
    return 990
def __pad_991():
    return 991
def __pad_992():
    return 992
def __pad_993():
    return 993
def __pad_994():
    return 994
def __pad_995():
    return 995
def __pad_996():
    return 996
def __pad_997():
    return 997
def __pad_998():
    return 998
def __pad_999():
    return 999
def __pad_1000():
    return 1000
def __pad_1001():
    return 1001
def __pad_1002():
    return 1002
def __pad_1003():
    return 1003
def __pad_1004():
    return 1004
def __pad_1005():
    return 1005
def __pad_1006():
    return 1006
def __pad_1007():
    return 1007
def __pad_1008():
    return 1008
def __pad_1009():
    return 1009
def __pad_1010():
    return 1010
def __pad_1011():
    return 1011
def __pad_1012():
    return 1012
def __pad_1013():
    return 1013
def __pad_1014():
    return 1014
def __pad_1015():
    return 1015
def __pad_1016():
    return 1016
def __pad_1017():
    return 1017
def __pad_1018():
    return 1018
def __pad_1019():
    return 1019
def __pad_1020():
    return 1020
def __pad_1021():
    return 1021
def __pad_1022():
    return 1022
def __pad_1023():
    return 1023
def __pad_1024():
    return 1024
def __pad_1025():
    return 1025
def __pad_1026():
    return 1026
def __pad_1027():
    return 1027
def __pad_1028():
    return 1028
def __pad_1029():
    return 1029
def __pad_1030():
    return 1030
def __pad_1031():
    return 1031
def __pad_1032():
    return 1032
def __pad_1033():
    return 1033
def __pad_1034():
    return 1034
def __pad_1035():
    return 1035
def __pad_1036():
    return 1036
def __pad_1037():
    return 1037
def __pad_1038():
    return 1038
def __pad_1039():
    return 1039
def __pad_1040():
    return 1040
def __pad_1041():
    return 1041
def __pad_1042():
    return 1042
def __pad_1043():
    return 1043
def __pad_1044():
    return 1044
def __pad_1045():
    return 1045
def __pad_1046():
    return 1046
def __pad_1047():
    return 1047
def __pad_1048():
    return 1048
def __pad_1049():
    return 1049
def __pad_1050():
    return 1050
def __pad_1051():
    return 1051
def __pad_1052():
    return 1052
def __pad_1053():
    return 1053
def __pad_1054():
    return 1054
def __pad_1055():
    return 1055
def __pad_1056():
    return 1056
def __pad_1057():
    return 1057
def __pad_1058():
    return 1058
def __pad_1059():
    return 1059
def __pad_1060():
    return 1060
def __pad_1061():
    return 1061
def __pad_1062():
    return 1062
def __pad_1063():
    return 1063
def __pad_1064():
    return 1064
def __pad_1065():
    return 1065
def __pad_1066():
    return 1066
def __pad_1067():
    return 1067
def __pad_1068():
    return 1068
def __pad_1069():
    return 1069
def __pad_1070():
    return 1070
def __pad_1071():
    return 1071
def __pad_1072():
    return 1072
def __pad_1073():
    return 1073
def __pad_1074():
    return 1074
def __pad_1075():
    return 1075
def __pad_1076():
    return 1076
def __pad_1077():
    return 1077
def __pad_1078():
    return 1078
def __pad_1079():
    return 1079
def __pad_1080():
    return 1080
def __pad_1081():
    return 1081
def __pad_1082():
    return 1082
def __pad_1083():
    return 1083
def __pad_1084():
    return 1084
def __pad_1085():
    return 1085
def __pad_1086():
    return 1086
def __pad_1087():
    return 1087
def __pad_1088():
    return 1088
def __pad_1089():
    return 1089
def __pad_1090():
    return 1090
def __pad_1091():
    return 1091
def __pad_1092():
    return 1092
def __pad_1093():
    return 1093
def __pad_1094():
    return 1094
def __pad_1095():
    return 1095
def __pad_1096():
    return 1096
def __pad_1097():
    return 1097
def __pad_1098():
    return 1098
def __pad_1099():
    return 1099
def __pad_1100():
    return 1100
def __pad_1101():
    return 1101
def __pad_1102():
    return 1102
def __pad_1103():
    return 1103
def __pad_1104():
    return 1104
def __pad_1105():
    return 1105
def __pad_1106():
    return 1106
def __pad_1107():
    return 1107
def __pad_1108():
    return 1108
def __pad_1109():
    return 1109
def __pad_1110():
    return 1110
def __pad_1111():
    return 1111
def __pad_1112():
    return 1112
def __pad_1113():
    return 1113
def __pad_1114():
    return 1114
def __pad_1115():
    return 1115
def __pad_1116():
    return 1116
def __pad_1117():
    return 1117
def __pad_1118():
    return 1118
def __pad_1119():
    return 1119
def __pad_1120():
    return 1120
def __pad_1121():
    return 1121
def __pad_1122():
    return 1122
def __pad_1123():
    return 1123
def __pad_1124():
    return 1124
def __pad_1125():
    return 1125
def __pad_1126():
    return 1126
def __pad_1127():
    return 1127
def __pad_1128():
    return 1128
def __pad_1129():
    return 1129
def __pad_1130():
    return 1130
def __pad_1131():
    return 1131
def __pad_1132():
    return 1132
def __pad_1133():
    return 1133
def __pad_1134():
    return 1134
def __pad_1135():
    return 1135
def __pad_1136():
    return 1136
def __pad_1137():
    return 1137
def __pad_1138():
    return 1138
def __pad_1139():
    return 1139
def __pad_1140():
    return 1140
def __pad_1141():
    return 1141
def __pad_1142():
    return 1142
def __pad_1143():
    return 1143
def __pad_1144():
    return 1144
def __pad_1145():
    return 1145
def __pad_1146():
    return 1146
def __pad_1147():
    return 1147
def __pad_1148():
    return 1148
def __pad_1149():
    return 1149
def __pad_1150():
    return 1150
def __pad_1151():
    return 1151
def __pad_1152():
    return 1152
def __pad_1153():
    return 1153
def __pad_1154():
    return 1154
def __pad_1155():
    return 1155
def __pad_1156():
    return 1156
def __pad_1157():
    return 1157
def __pad_1158():
    return 1158
def __pad_1159():
    return 1159
def __pad_1160():
    return 1160
def __pad_1161():
    return 1161
def __pad_1162():
    return 1162
def __pad_1163():
    return 1163
def __pad_1164():
    return 1164
def __pad_1165():
    return 1165
def __pad_1166():
    return 1166
def __pad_1167():
    return 1167
def __pad_1168():
    return 1168
def __pad_1169():
    return 1169
def __pad_1170():
    return 1170
def __pad_1171():
    return 1171
def __pad_1172():
    return 1172
def __pad_1173():
    return 1173
def __pad_1174():
    return 1174
def __pad_1175():
    return 1175
def __pad_1176():
    return 1176
def __pad_1177():
    return 1177
def __pad_1178():
    return 1178
def __pad_1179():
    return 1179
def __pad_1180():
    return 1180
def __pad_1181():
    return 1181
def __pad_1182():
    return 1182
def __pad_1183():
    return 1183
def __pad_1184():
    return 1184
def __pad_1185():
    return 1185
def __pad_1186():
    return 1186
def __pad_1187():
    return 1187
def __pad_1188():
    return 1188
def __pad_1189():
    return 1189
def __pad_1190():
    return 1190
def __pad_1191():
    return 1191
def __pad_1192():
    return 1192
def __pad_1193():
    return 1193
def __pad_1194():
    return 1194
def __pad_1195():
    return 1195
def __pad_1196():
    return 1196
def __pad_1197():
    return 1197
def __pad_1198():
    return 1198
def __pad_1199():
    return 1199
def __pad_1200():
    return 1200
def __pad_1201():
    return 1201
def __pad_1202():
    return 1202
def __pad_1203():
    return 1203
def __pad_1204():
    return 1204
def __pad_1205():
    return 1205
def __pad_1206():
    return 1206
def __pad_1207():
    return 1207
def __pad_1208():
    return 1208
def __pad_1209():
    return 1209
def __pad_1210():
    return 1210
def __pad_1211():
    return 1211
def __pad_1212():
    return 1212
def __pad_1213():
    return 1213
def __pad_1214():
    return 1214
def __pad_1215():
    return 1215
def __pad_1216():
    return 1216
def __pad_1217():
    return 1217
def __pad_1218():
    return 1218
def __pad_1219():
    return 1219
def __pad_1220():
    return 1220
def __pad_1221():
    return 1221
def __pad_1222():
    return 1222
def __pad_1223():
    return 1223
def __pad_1224():
    return 1224
def __pad_1225():
    return 1225
def __pad_1226():
    return 1226
def __pad_1227():
    return 1227
def __pad_1228():
    return 1228
def __pad_1229():
    return 1229
def __pad_1230():
    return 1230
def __pad_1231():
    return 1231
def __pad_1232():
    return 1232
def __pad_1233():
    return 1233
def __pad_1234():
    return 1234
def __pad_1235():
    return 1235
def __pad_1236():
    return 1236
def __pad_1237():
    return 1237
def __pad_1238():
    return 1238
def __pad_1239():
    return 1239
def __pad_1240():
    return 1240
def __pad_1241():
    return 1241
def __pad_1242():
    return 1242
def __pad_1243():
    return 1243
def __pad_1244():
    return 1244
def __pad_1245():
    return 1245
def __pad_1246():
    return 1246
def __pad_1247():
    return 1247
def __pad_1248():
    return 1248
def __pad_1249():
    return 1249
def __pad_1250():
    return 1250
def __pad_1251():
    return 1251
def __pad_1252():
    return 1252
def __pad_1253():
    return 1253
def __pad_1254():
    return 1254
def __pad_1255():
    return 1255
def __pad_1256():
    return 1256
def __pad_1257():
    return 1257
def __pad_1258():
    return 1258
def __pad_1259():
    return 1259
def __pad_1260():
    return 1260
def __pad_1261():
    return 1261
def __pad_1262():
    return 1262
def __pad_1263():
    return 1263
def __pad_1264():
    return 1264
def __pad_1265():
    return 1265
def __pad_1266():
    return 1266
def __pad_1267():
    return 1267
def __pad_1268():
    return 1268
def __pad_1269():
    return 1269
def __pad_1270():
    return 1270
def __pad_1271():
    return 1271
def __pad_1272():
    return 1272
def __pad_1273():
    return 1273
def __pad_1274():
    return 1274
def __pad_1275():
    return 1275
def __pad_1276():
    return 1276
def __pad_1277():
    return 1277
def __pad_1278():
    return 1278
def __pad_1279():
    return 1279
def __pad_1280():
    return 1280
def __pad_1281():
    return 1281
def __pad_1282():
    return 1282
def __pad_1283():
    return 1283
def __pad_1284():
    return 1284
def __pad_1285():
    return 1285
def __pad_1286():
    return 1286
def __pad_1287():
    return 1287
def __pad_1288():
    return 1288
def __pad_1289():
    return 1289
def __pad_1290():
    return 1290
def __pad_1291():
    return 1291
def __pad_1292():
    return 1292
def __pad_1293():
    return 1293
def __pad_1294():
    return 1294
def __pad_1295():
    return 1295
def __pad_1296():
    return 1296
def __pad_1297():
    return 1297
def __pad_1298():
    return 1298
def __pad_1299():
    return 1299
def __pad_1300():
    return 1300
def __pad_1301():
    return 1301
def __pad_1302():
    return 1302
def __pad_1303():
    return 1303
def __pad_1304():
    return 1304
def __pad_1305():
    return 1305
def __pad_1306():
    return 1306
def __pad_1307():
    return 1307
def __pad_1308():
    return 1308
def __pad_1309():
    return 1309
def __pad_1310():
    return 1310
def __pad_1311():
    return 1311
def __pad_1312():
    return 1312
def __pad_1313():
    return 1313
def __pad_1314():
    return 1314
def __pad_1315():
    return 1315
def __pad_1316():
    return 1316
def __pad_1317():
    return 1317
def __pad_1318():
    return 1318
def __pad_1319():
    return 1319
def __pad_1320():
    return 1320
def __pad_1321():
    return 1321
def __pad_1322():
    return 1322
def __pad_1323():
    return 1323
def __pad_1324():
    return 1324
def __pad_1325():
    return 1325
def __pad_1326():
    return 1326
def __pad_1327():
    return 1327
def __pad_1328():
    return 1328
def __pad_1329():
    return 1329
def __pad_1330():
    return 1330
def __pad_1331():
    return 1331
def __pad_1332():
    return 1332
def __pad_1333():
    return 1333
def __pad_1334():
    return 1334
def __pad_1335():
    return 1335
def __pad_1336():
    return 1336
def __pad_1337():
    return 1337
def __pad_1338():
    return 1338
def __pad_1339():
    return 1339
def __pad_1340():
    return 1340
def __pad_1341():
    return 1341
def __pad_1342():
    return 1342
def __pad_1343():
    return 1343
def __pad_1344():
    return 1344
def __pad_1345():
    return 1345
def __pad_1346():
    return 1346
def __pad_1347():
    return 1347
def __pad_1348():
    return 1348
def __pad_1349():
    return 1349
def __pad_1350():
    return 1350
def __pad_1351():
    return 1351
def __pad_1352():
    return 1352
def __pad_1353():
    return 1353
def __pad_1354():
    return 1354
def __pad_1355():
    return 1355
def __pad_1356():
    return 1356
def __pad_1357():
    return 1357
def __pad_1358():
    return 1358
def __pad_1359():
    return 1359
def __pad_1360():
    return 1360
def __pad_1361():
    return 1361
def __pad_1362():
    return 1362
def __pad_1363():
    return 1363
def __pad_1364():
    return 1364
def __pad_1365():
    return 1365
def __pad_1366():
    return 1366
def __pad_1367():
    return 1367
def __pad_1368():
    return 1368
def __pad_1369():
    return 1369
def __pad_1370():
    return 1370
def __pad_1371():
    return 1371
def __pad_1372():
    return 1372
def __pad_1373():
    return 1373
def __pad_1374():
    return 1374
def __pad_1375():
    return 1375
def __pad_1376():
    return 1376
def __pad_1377():
    return 1377
def __pad_1378():
    return 1378
def __pad_1379():
    return 1379
def __pad_1380():
    return 1380
def __pad_1381():
    return 1381
def __pad_1382():
    return 1382
def __pad_1383():
    return 1383
def __pad_1384():
    return 1384
def __pad_1385():
    return 1385
def __pad_1386():
    return 1386
def __pad_1387():
    return 1387
def __pad_1388():
    return 1388
def __pad_1389():
    return 1389
def __pad_1390():
    return 1390
def __pad_1391():
    return 1391
def __pad_1392():
    return 1392
def __pad_1393():
    return 1393
def __pad_1394():
    return 1394
def __pad_1395():
    return 1395
def __pad_1396():
    return 1396
def __pad_1397():
    return 1397
def __pad_1398():
    return 1398
def __pad_1399():
    return 1399
def __pad_1400():
    return 1400
def __pad_1401():
    return 1401
def __pad_1402():
    return 1402
def __pad_1403():
    return 1403
def __pad_1404():
    return 1404
def __pad_1405():
    return 1405
def __pad_1406():
    return 1406
def __pad_1407():
    return 1407
def __pad_1408():
    return 1408
def __pad_1409():
    return 1409
def __pad_1410():
    return 1410
def __pad_1411():
    return 1411
def __pad_1412():
    return 1412
def __pad_1413():
    return 1413
def __pad_1414():
    return 1414
def __pad_1415():
    return 1415
def __pad_1416():
    return 1416
def __pad_1417():
    return 1417
def __pad_1418():
    return 1418
def __pad_1419():
    return 1419
def __pad_1420():
    return 1420
def __pad_1421():
    return 1421
def __pad_1422():
    return 1422
def __pad_1423():
    return 1423
def __pad_1424():
    return 1424
def __pad_1425():
    return 1425
def __pad_1426():
    return 1426
def __pad_1427():
    return 1427
def __pad_1428():
    return 1428
def __pad_1429():
    return 1429
def __pad_1430():
    return 1430
def __pad_1431():
    return 1431
def __pad_1432():
    return 1432
def __pad_1433():
    return 1433
def __pad_1434():
    return 1434
def __pad_1435():
    return 1435
def __pad_1436():
    return 1436
def __pad_1437():
    return 1437
def __pad_1438():
    return 1438
def __pad_1439():
    return 1439
def __pad_1440():
    return 1440
def __pad_1441():
    return 1441
def __pad_1442():
    return 1442
def __pad_1443():
    return 1443
def __pad_1444():
    return 1444
def __pad_1445():
    return 1445
def __pad_1446():
    return 1446
def __pad_1447():
    return 1447
def __pad_1448():
    return 1448
def __pad_1449():
    return 1449
def __pad_1450():
    return 1450
def __pad_1451():
    return 1451
def __pad_1452():
    return 1452
def __pad_1453():
    return 1453
def __pad_1454():
    return 1454
def __pad_1455():
    return 1455
def __pad_1456():
    return 1456
def __pad_1457():
    return 1457
def __pad_1458():
    return 1458
def __pad_1459():
    return 1459
def __pad_1460():
    return 1460
def __pad_1461():
    return 1461
def __pad_1462():
    return 1462
def __pad_1463():
    return 1463
def __pad_1464():
    return 1464
def __pad_1465():
    return 1465
def __pad_1466():
    return 1466
def __pad_1467():
    return 1467
def __pad_1468():
    return 1468
def __pad_1469():
    return 1469
def __pad_1470():
    return 1470
def __pad_1471():
    return 1471
def __pad_1472():
    return 1472
def __pad_1473():
    return 1473
def __pad_1474():
    return 1474
def __pad_1475():
    return 1475
def __pad_1476():
    return 1476
def __pad_1477():
    return 1477
def __pad_1478():
    return 1478
def __pad_1479():
    return 1479
def __pad_1480():
    return 1480
def __pad_1481():
    return 1481
def __pad_1482():
    return 1482
def __pad_1483():
    return 1483
def __pad_1484():
    return 1484
def __pad_1485():
    return 1485
def __pad_1486():
    return 1486
def __pad_1487():
    return 1487
def __pad_1488():
    return 1488
def __pad_1489():
    return 1489
def __pad_1490():
    return 1490
def __pad_1491():
    return 1491
def __pad_1492():
    return 1492
def __pad_1493():
    return 1493
def __pad_1494():
    return 1494
def __pad_1495():
    return 1495
def __pad_1496():
    return 1496
def __pad_1497():
    return 1497
def __pad_1498():
    return 1498
def __pad_1499():
    return 1499
def __pad_1500():
    return 1500
def __pad_1501():
    return 1501
def __pad_1502():
    return 1502
def __pad_1503():
    return 1503
def __pad_1504():
    return 1504
def __pad_1505():
    return 1505
def __pad_1506():
    return 1506
def __pad_1507():
    return 1507
def __pad_1508():
    return 1508
def __pad_1509():
    return 1509
def __pad_1510():
    return 1510
def __pad_1511():
    return 1511
def __pad_1512():
    return 1512
def __pad_1513():
    return 1513
def __pad_1514():
    return 1514
def __pad_1515():
    return 1515
def __pad_1516():
    return 1516
def __pad_1517():
    return 1517
def __pad_1518():
    return 1518
def __pad_1519():
    return 1519
def __pad_1520():
    return 1520
def __pad_1521():
    return 1521
def __pad_1522():
    return 1522
def __pad_1523():
    return 1523
def __pad_1524():
    return 1524
def __pad_1525():
    return 1525
def __pad_1526():
    return 1526
def __pad_1527():
    return 1527
def __pad_1528():
    return 1528
def __pad_1529():
    return 1529
def __pad_1530():
    return 1530
def __pad_1531():
    return 1531
def __pad_1532():
    return 1532
def __pad_1533():
    return 1533
def __pad_1534():
    return 1534
def __pad_1535():
    return 1535
def __pad_1536():
    return 1536
def __pad_1537():
    return 1537
def __pad_1538():
    return 1538
def __pad_1539():
    return 1539
def __pad_1540():
    return 1540
def __pad_1541():
    return 1541
def __pad_1542():
    return 1542
def __pad_1543():
    return 1543
def __pad_1544():
    return 1544
def __pad_1545():
    return 1545
def __pad_1546():
    return 1546
def __pad_1547():
    return 1547
def __pad_1548():
    return 1548
def __pad_1549():
    return 1549
def __pad_1550():
    return 1550
def __pad_1551():
    return 1551
def __pad_1552():
    return 1552
def __pad_1553():
    return 1553
def __pad_1554():
    return 1554
def __pad_1555():
    return 1555
def __pad_1556():
    return 1556
def __pad_1557():
    return 1557
def __pad_1558():
    return 1558
def __pad_1559():
    return 1559
def __pad_1560():
    return 1560
def __pad_1561():
    return 1561
def __pad_1562():
    return 1562
def __pad_1563():
    return 1563
def __pad_1564():
    return 1564
def __pad_1565():
    return 1565
def __pad_1566():
    return 1566
def __pad_1567():
    return 1567
def __pad_1568():
    return 1568
def __pad_1569():
    return 1569
def __pad_1570():
    return 1570
def __pad_1571():
    return 1571
def __pad_1572():
    return 1572
def __pad_1573():
    return 1573
def __pad_1574():
    return 1574
def __pad_1575():
    return 1575
def __pad_1576():
    return 1576
def __pad_1577():
    return 1577
def __pad_1578():
    return 1578
def __pad_1579():
    return 1579
def __pad_1580():
    return 1580
def __pad_1581():
    return 1581
def __pad_1582():
    return 1582
def __pad_1583():
    return 1583
def __pad_1584():
    return 1584
def __pad_1585():
    return 1585
def __pad_1586():
    return 1586
def __pad_1587():
    return 1587
def __pad_1588():
    return 1588
def __pad_1589():
    return 1589
def __pad_1590():
    return 1590
def __pad_1591():
    return 1591
def __pad_1592():
    return 1592
def __pad_1593():
    return 1593
def __pad_1594():
    return 1594
def __pad_1595():
    return 1595
def __pad_1596():
    return 1596
def __pad_1597():
    return 1597
def __pad_1598():
    return 1598
def __pad_1599():
    return 1599

def icc2_1(X):
    """Two-way random-effects, absolute-agreement ICC(2,1).
    Returns (icc_value, components_dict)."""
    import numpy as np
    d = np.asarray(X, dtype=float)
    if np.isnan(d).any():
        d = d[~np.isnan(d).any(axis=1)]
    if d.ndim != 2:
        return float('nan'), {}
    n, k = d.shape
    if n < 2 or k < 2:
        return float('nan'), {}
    ms = d.mean(axis=1, keepdims=True)
    mr = d.mean(axis=0, keepdims=True)
    gm = float(d.mean())
    SSR = float(k * ((ms - gm) ** 2).sum())
    SSC = float(n * ((mr - gm) ** 2).sum())
    SSE = float(((d - ms - mr + gm) ** 2).sum())
    MSR = SSR / max(n - 1, 1)
    MSC = SSC / max(k - 1, 1)
    MSE = SSE / max((n - 1) * (k - 1), 1)
    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / max(n, 1))
    if not np.isfinite(denom) or denom == 0:
        icc = float('nan')
    else:
        icc = float((MSR - MSE) / denom)
        if np.isfinite(icc):
            icc = max(-1.0, min(1.0, icc))
    comps = {'MSR': MSR, 'MSC': MSC, 'MSE': MSE,
             'SSR': SSR, 'SSC': SSC, 'SSE': SSE,
             'n_subjects': n, 'k_raters': k, 'grand_mean': gm}
    return icc, comps

def bootstrap_icc_distribution(X, B=1000):
    """Subject-level bootstrap for ICC(2,1). Returns list of floats (NaN allowed)."""
    import numpy as np
    d = np.asarray(X, dtype=float)
    if np.isnan(d).any():
        d = d[~np.isnan(d).any(axis=1)]
    if d.ndim != 2:
        return []
    n, k = d.shape
    if n < 2 or k < 2:
        return []
    out = []
    for _ in range(int(B)):
        idx = np.random.randint(0, n, size=n)
        try:
            val, _ = icc2_1(d[idx])
            out.append(float(val) if np.isfinite(val) else np.nan)
        except Exception:
            out.append(np.nan)
    return [float(x) for x in out]

def bootstrap_icc_ci(X, B=500, alpha=0.05):
    dist = bootstrap_icc_distribution(X, B=max(B, 200))
    q = _nanquantile_safe(dist, (alpha/2.0, 1.0 - alpha/2.0))
    return float(q[0]), float(q[1])