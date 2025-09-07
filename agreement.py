import math
try:
    import numpy as _np
except Exception:
    _np=None
try:
    from sklearn.metrics import roc_curve as _roc_curve, auc as _auc, precision_recall_curve as _pr_curve, average_precision_score as _ap
except Exception:
    _roc_curve=_auc=_pr_curve=_ap=None
def _wilson_ci(success,total,z=1.96):
    if not total or success is None: return float('nan'),float('nan')
    p=float(success)/float(total); denom=1.0+(z*z)/total; centre=p+(z*z)/(2*total)
    adj=z*math.sqrt((p*(1-p)+(z*z)/(4*total))/total)
    lo=(centre-adj)/denom; hi=(centre+adj)/denom
    lo=0.0 if lo<0 else (1.0 if lo>1 else lo); hi=0.0 if hi<0 else (1.0 if hi>1 else hi)
    return float(lo),float(hi)
def _safe_div(a,b): return float(a)/float(b) if b else float('nan')
def binary_metrics(y_true,y_pred):
    if len(y_true)!=len(y_pred): raise ValueError('y_true and y_pred must be same length')
    tp=fp=tn=fn=0
    for t,p in zip(y_true,y_pred):
        if t==1 and p==1: tp+=1
        elif t==0 and p==1: fp+=1
        elif t==0 and p==0: tn+=1
        elif t==1 and p==0: fn+=1
    N=tp+fp+tn+fn; acc=_safe_div(tp+tn,N); sens=_safe_div(tp,tp+fn); spec=_safe_div(tn,tn+fp)
    ppv=_safe_div(tp,tp+fp); npv=_safe_div(tn,tn+fn); f1=_safe_div(2*tp,2*tp+fp+fn)
    bal=0.5*(sens+spec) if not math.isnan(sens) and not math.isnan(spec) else float('nan')
    yj=(sens+spec-1.0) if not math.isnan(sens) and not math.isnan(spec) else float('nan')
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc=_safe_div(tp*tn-fp*fn, math.sqrt(denom)) if denom else float('nan')
    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'N':N,'Accuracy':acc,'Accuracy_CI':_wilson_ci(tp+tn,N),'Sensitivity_TPR':sens,'Sensitivity_CI':_wilson_ci(tp,tp+fn),'Specificity_TNR':spec,'Specificity_CI':_wilson_ci(tn,tn+fp),'PPV':ppv,'PPV_CI':_wilson_ci(tp,tp+fp),'NPV':npv,'NPV_CI':_wilson_ci(tn,tn+fn),'F1':f1,'Balanced_Accuracy':bal,'Youdens_J':yj,'MCC':mcc}
def roc_points_auc(y_true,scores):
    if _roc_curve is None or _auc is None: raise ImportError('scikit-learn not available for ROC')
    fpr,tpr,_=_roc_curve(y_true,scores); return fpr.tolist(),tpr.tolist(),float(_auc(fpr,tpr))
def pr_points_ap(y_true,scores):
    if _pr_curve is None or _ap is None: raise ImportError('scikit-learn not available for PR')
    precision,recall,_=_pr_curve(y_true,scores); ap=_ap(y_true,scores); return recall.tolist(),precision.tolist(),float(ap)
def fleiss_kappa_from_raw(matrix):
    if _np is None: raise ImportError('numpy required')
    cats=sorted({str(l).strip() for row in matrix for l in row if l is not None and str(l).strip()!=''})
    if not cats: return float('nan'),[],{}
    c2i={c:i for i,c in enumerate(cats)}; counts=[]
    for row in matrix:
        c=_np.zeros(len(cats),dtype=float)
        for lab in row:
            if lab is None: continue
            s=str(lab).strip()
            if s=='': continue
            c[c2i[s]]+=1.0
        counts.append(c)
    counts=_np.asarray(counts); valid=_np.array([c.sum()>=2 for c in counts])
    if not valid.any(): return float('nan'),[],{}
    counts=counts[valid]; n_i=counts.sum(axis=1)
    P_i=((counts*(counts-1)).sum(axis=1))/(n_i*(n_i-1))
    p_j=counts.sum(axis=0)/n_i.sum(); Pbar=float(_np.nanmean(P_i)); Pe=float((p_j**2).sum())
    kappa=(Pbar-Pe)/(1-Pe) if (1-Pe)!=0 else float('nan')
    marg={cats[i]:float(p_j[i]) for i in range(len(cats))}
    return float(kappa),P_i.tolist(),marg
def icc2_1(data):
    if _np is None: raise ImportError('numpy required')
    d=_np.asarray(data,dtype=float); n,k=d.shape
    ms=d.mean(axis=1,keepdims=True); mr=d.mean(axis=0,keepdims=True); gm=d.mean()
    MSR=k*((ms-gm)**2).sum()/(n-1); MSC=n*((mr-gm)**2).sum()/(k-1)
    MSE=((d-ms-mr+gm)**2).sum()/((n-1)*(k-1))
    return float((MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n))
def bootstrap_icc_ci(data,B=500,alpha=0.05):
    if _np is None: raise ImportError('numpy required')
    d=_np.asarray(data,dtype=float); n=d.shape[0]; vals=[]
    for _ in range(B):
        idx=_np.random.randint(0,n,size=n)
        try: vals.append(icc2_1(d[idx]))
        except Exception: pass
    if not vals: return float('nan'),float('nan')
    vals.sort(); lo=vals[int((alpha/2)*len(vals))]; hi=vals[int((1-alpha/2)*len(vals))-1]
    return float(lo),float(hi)
