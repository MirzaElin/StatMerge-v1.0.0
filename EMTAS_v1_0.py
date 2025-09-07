import os, json, math, csv, datetime
import numpy as np
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError as e:
        print("\n[ERROR] Qt bindings not found. Install one of these:")
        print("  pip install PyQt5 matplotlib numpy")
        print("  (or) pip install PySide6 matplotlib numpy")
        raise

import matplotlib.pyplot as plt


from metrics import (
    binary_metrics, roc_points_auc, pr_points_ap,
    fleiss_kappa_from_raw, bootstrap_fleiss_ci,
    icc2_1, bootstrap_icc_ci, bootstrap_icc_distribution
)

APP_TITLE = "EMTAS v1.0 - © 2025 Mirza Niaz Zaman Elin. All rights reserved."
EXPORT_DIR = "exports"


def ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)




def spearman_brown(icc1, k):
    arr = np.asarray(icc1, dtype=float)
    if k <= 1:
        return arr
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (k * arr) / (1.0 + (k - 1.0) * arr)
    return out


def icc_avg(icc1, k, reduce='mean'):

   def confusion_counts(y_true, y_pred):
    """Return TP, FP, TN, FN for binary 0/1 labels."""
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1: tp += 1
        elif yt == 0 and yp == 1: fp += 1
        elif yt == 0 and yp == 0: tn += 1
        elif yt == 1 and yp == 0: fn += 1
    return tp, fp, tn, fn
    sb = spearman_brown(icc1, k)
    if np.isscalar(sb) or (isinstance(sb, np.ndarray) and sb.ndim == 0):
        return float(sb)
    if reduce == 'mean':
        return float(np.nanmean(sb))
    elif reduce == 'median':
        return float(np.nanmedian(sb))
    else:
        return sb




def _as_clean_labels(seq):
    """Convert values to strings, keep None for empties."""
    out = []
    for x in seq:
        if x is None:
            out.append(None)
        else:
            s = str(x).strip()
            out.append(None if s == '' else s)
    return out


def cohens_kappa(labels1, labels2):
    """Cohen's kappa for two raters, nominal categories.
    labels1/labels2: sequences of strings/ints; None/'' ignored.
    """
    a = _as_clean_labels(labels1)
    b = _as_clean_labels(labels2)
    pairs = [(x, y) for x, y in zip(a, b) if (x is not None and y is not None)]
    if not pairs:
        return float('nan'), {
            'po': float('nan'), 'pe': float('nan'), 'n': 0,
            'categories': {}, 'confusion': {}
        }
    cats = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in pairs:
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n if n else float('nan')
    row = m.sum(axis=1) / n
    col = m.sum(axis=0) / n
    pe = float((row @ col))
    denom = 1 - pe
    kappa = (po - pe) / denom if denom != 0 else float('nan')
    
    conf = {cats[i]: {cats[j]: int(m[i, j]) for j in range(len(cats))} for i in range(len(cats))}
    cat_freq = {c: int(m[i, :].sum()) for c, i in idx.items()}
    return float(kappa), {'po': float(po), 'pe': float(pe), 'n': int(n), 'categories': cat_freq, 'confusion': conf}


def krippendorff_alpha_nominal(table):
    """Krippendorff's alpha (nominal) for 2+ raters with missing values.
    `table` is list-of-lists per unit: [rating_by_rater1, rating_by_rater2, ...].
    Missing values: None or ''. Alpha is 1 - Do/De using pair disagreement.
    """
    
    units = []
    for row in table:
        cleaned = [None if (v is None or str(v).strip() == '') else str(v).strip() for v in row]
        
        if sum(1 for v in cleaned if v is not None) >= 2:
            units.append(cleaned)
    if not units:
        return float('nan'), {'Do': float('nan'), 'De': float('nan'), 'total_pairs': 0, 'category_frequency': {}}

    
    cat_counts = {}
    total_ratings = 0
    total_pairs = 0
    disagree_pairs = 0

    for row in units:
        vals = [v for v in row if v is not None]
        total_ratings += len(vals)
        for v in vals:
            cat_counts[v] = cat_counts.get(v, 0) + 1
        
        n_i = len(vals)
        total_pairs += n_i * (n_i - 1)
        agree_pairs_i = 0
        
        from collections import Counter
        cts = Counter(vals)
        for c, n_ic in cts.items():
            agree_pairs_i += n_ic * (n_ic - 1)
        disagree_pairs += (n_i * (n_i - 1) - agree_pairs_i)

    if total_pairs == 0:
        return float('nan'), {'Do': float('nan'), 'De': float('nan'), 'total_pairs': 0, 'category_frequency': cat_counts}

    Do = disagree_pairs / total_pairs
    if total_ratings == 0:
        return float('nan'), {'Do': float('nan'), 'De': float('nan'), 'total_pairs': total_pairs, 'category_frequency': cat_counts}

    
    probs = [cnt / total_ratings for cnt in cat_counts.values()]
    De = 1.0 - float(np.sum(np.square(probs)))

    if De == 0:
        alpha = float('nan')  
    else:
        alpha = 1.0 - (Do / De)

    return float(alpha), {'Do': float(Do), 'De': float(De), 'total_pairs': int(total_pairs), 'category_frequency': cat_counts}




class LoginDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign in")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)
        self.user = QtWidgets.QLineEdit(self)
        self.passw = QtWidgets.QLineEdit(self); self.passw.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addRow("Username", self.user)
        layout.addRow("Password", self.passw)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)
    def get_user(self):
        return self.user.text().strip() or "User"




class BinaryTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name/Group:"))
        self.name_edit = QtWidgets.QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g., Group-1, Team A, Rater X")
        name_row.addWidget(self.name_edit)

        self.table = QtWidgets.QTableWidget(10, 4, self)
        self.table.setHorizontalHeaderLabels(["subject_id","true_label","pred_label","score"])
        self.result = QtWidgets.QPlainTextEdit(self); self.result.setReadOnly(True)

        
        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.graph_btn = QtWidgets.QPushButton("Graph")
        self.save_graph_btn = QtWidgets.QPushButton("Save Graph")
        self.save_btn = QtWidgets.QPushButton("Save Data/JSON")
        self.export_results_btn = QtWidgets.QPushButton("Export Results (.txt)")
        self.import_btn = QtWidgets.QPushButton("Import CSV")
        self.batch_btn = QtWidgets.QPushButton("Batch from Folder")
        self.add_one_row_btn = QtWidgets.QPushButton("Add 1 item")
        self.rem_one_row_btn = QtWidgets.QPushButton("Remove 1 item")
        self.add_row_btn = QtWidgets.QPushButton("Add 10 rows")
        self.load_tpl = QtWidgets.QPushButton("Load template")

        btn_row = QtWidgets.QHBoxLayout()
        for b in [self.calc_btn, self.graph_btn, self.save_graph_btn, self.save_btn,
                  self.export_results_btn, self.import_btn, self.batch_btn,
                  self.add_one_row_btn, self.rem_one_row_btn, self.add_row_btn, self.load_tpl]:
            btn_row.addWidget(b)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(name_row)
        layout.addWidget(self.table)
        layout.addLayout(btn_row)
        layout.addWidget(self.result)

        
        self.calc_btn.clicked.connect(self.calculate)
        self.graph_btn.clicked.connect(self.graph)
        self.save_graph_btn.clicked.connect(self.save_graph)
        self.save_btn.clicked.connect(self.save)
        self.export_results_btn.clicked.connect(self.export_results)
        self.import_btn.clicked.connect(self.import_csv)
        self.batch_btn.clicked.connect(self.batch_from_folder)
        self.add_one_row_btn.clicked.connect(self.add_one_row)
        self.rem_one_row_btn.clicked.connect(self.rem_one_row)
        self.add_row_btn.clicked.connect(self.add_rows)
        self.load_tpl.clicked.connect(self.load_template)
        self.last_calc = None

    
    def load_template(self):
        path = os.path.join("examples","binary_template.csv")
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self,"Template","Template not found.")
            return
        self._load_csv_to_table(path)

    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_csv_to_table(path)

    def _load_csv_to_table(self, path):
        try:
            with open(path, "r", newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)
            
            if header and len(header) >= 3:
                
                cols = ["subject_id","true_label","pred_label","score"]
                
                self.table.setColumnCount(len(cols))
                self.table.setHorizontalHeaderLabels(cols)
                self.table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    for j in range(min(len(cols), len(row))):
                        self.table.setItem(i, j, QtWidgets.QTableWidgetItem(row[j]))
            else:
                
                self.table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    for j in range(min(4, len(row))):
                        self.table.setItem(i, j, QtWidgets.QTableWidgetItem(row[j]))
            QtWidgets.QMessageBox.information(self, "Import", f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))

    def export_results(self):
        ensure_export_dir()
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"agreement_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"icc_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"binary_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception as _e:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (self.name_edit.text().strip() or "Group").replace(' ', '_')
        txt_path = os.path.join(EXPORT_DIR, f"binary_results_{name}_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(txt_path)}")

    
    def add_rows(self):
        self.table.setRowCount(self.table.rowCount()+10)

    def add_one_row(self):
        self.table.setRowCount(self.table.rowCount()+1)

    def rem_one_row(self):
        if self.table.rowCount() > 1:
            self.table.setRowCount(self.table.rowCount()-1)

    
    def _collect(self):
        y_true, y_pred, scores = [], [], []
        for r in range(self.table.rowCount()):
            t_item = self.table.item(r,1); p_item = self.table.item(r,2)
            if not t_item or not p_item:
                continue
            t = t_item.text().strip(); p = p_item.text().strip()
            if t=="" or p=="":
                continue
            try:
                t = int(float(t)); p = int(float(p))
            except:
                continue
            y_true.append(t); y_pred.append(p)
            s_item = self.table.item(r,3)
            if s_item and s_item.text().strip()!="":
                try:
                    scores.append(float(s_item.text().strip()))
                except:
                    scores.append(None)
            else:
                scores.append(None)
        score_vals = [s for s in scores if isinstance(s,(int,float))]
        return y_true, y_pred, score_vals if len(score_vals)==len(y_true) else []

    
    def calculate(self):
        y_true, y_pred, scores = self._collect()
        if not y_true:
            QtWidgets.QMessageBox.warning(self,"Input","No valid rows.")
            return
        res = binary_metrics(y_true, y_pred)
        lines = []
        name = self.name_edit.text().strip()
        if name:
            lines.append(f"Label: {name}")
            lines.append("")
        for k in ["TP","FP","TN","FN","N"]:
            lines.append(f"{k}: {res[k]}")
        lines.append("")
        def fmt_pct(x):
            return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{x*100:.2f}%"
        lines.append(f"Accuracy: {fmt_pct(res['Accuracy'])}  CI [{fmt_pct(res['Accuracy_CI'][0])}, {fmt_pct(res['Accuracy_CI'][1])}]")
        lines.append(f"Sensitivity (TPR): {fmt_pct(res['Sensitivity_TPR'])}  CI [{fmt_pct(res['Sensitivity_CI'][0])}, {fmt_pct(res['Sensitivity_CI'][1])}]")
        lines.append(f"Specificity (TNR): {fmt_pct(res['Specificity_TNR'])}  CI [{fmt_pct(res['Specificity_CI'][0])}, {fmt_pct(res['Specificity_CI'][1])}]")
        lines.append(f"PPV: {fmt_pct(res['PPV'])}  CI [{fmt_pct(res['PPV_CI'][0])}, {fmt_pct(res['PPV_CI'][1])}]")
        lines.append(f"NPV: {fmt_pct(res['NPV'])}  CI [{fmt_pct(res['NPV_CI'][0])}, {fmt_pct(res['NPV_CI'][1])}]")
        lines.append(f"F1: {fmt_pct(res['F1'])}")
        lines.append(f"Balanced Accuracy: {fmt_pct(res['Balanced_Accuracy'])}")
        yj = res['Youdens_J']; yj_str = 'nan' if math.isnan(yj) else f"{yj:.4f}"
        mcc = res['MCC']; mcc_str = 'nan' if math.isnan(mcc) else f"{mcc:.4f}"
        lines.append(f"Youden's J: {yj_str}")
        lines.append(f"MCC: {mcc_str}")
        if scores and len(set(scores))>1:
            xs, ys, auc = roc_points_auc(y_true, scores)
            r, p, ap = pr_points_ap(y_true, scores)
            lines.append(f"AUC (ROC): {auc:.4f}")
            lines.append(f"Average Precision (PR AUC): {ap:.4f}")
        else:
            lines.append("ROC/PR disabled (provide Score column with ≥2 unique values).")
        self.result.setPlainText("\n".join(lines))
        self.last_calc = {"results": res, "name": name}

    def graph(self):
        try:
            if self.last_calc is None:
                self.calculate()
                if self.last_calc is None:
                    return
            y_true, y_pred, scores = self._collect()
            if scores and len(set(scores))>1:
                xs, ys, auc = roc_points_auc(y_true, scores)
                self._last_graph_kind = 'rocpr'
                fig = plt.figure()
                plt.plot(xs, ys, label=f"AUC={auc:.3f}")
                plt.plot([0,1],[0,1],'--')
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
                plt.legend(loc="lower right")
                self._show_fig(fig)
                r, p, ap = pr_points_ap(y_true, scores)
                self._last_graph_kind = 'pr'
                fig2 = plt.figure()
                plt.step(r, p, where="post", label=f"AP={ap:.3f}")
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
                plt.legend(loc="lower left")
                self._show_fig(fig2)
            else:
                QtWidgets.QMessageBox.information(self, "ROC/PR unavailable", "ROC/PR curves require a numeric 'score' column with at least 2 unique values. Showing confusion matrix instead.")
                tp, fp, tn, fn = confusion_counts(y_true, y_pred)
                mat = np.array([[tn, fp],[fn, tp]])
                fig = plt.figure()
                plt.imshow(mat, interpolation="nearest")
                for (i,j), val in np.ndenumerate(mat):
                    plt.text(j, i, str(val), ha="center", va="center")
                plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
                plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
                self._show_fig(fig)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Graph error", str(e))

    def _show_fig(self, fig):
        if not hasattr(self, '_fig_windows'):
            self._fig_windows = []
        w = FigureWindow(fig)
        self._fig_windows.append(w)
        w.show()

    def save_graph(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (self.name_edit.text().strip() or "Group").replace(' ', '_')
        path = os.path.join(EXPORT_DIR, f"binary_graph_{name}_{ts}.png")
        y_true, y_pred, scores = self._collect()
        if scores and len(set(scores))>1:
            xs, ys, auc = roc_points_auc(y_true, scores)
            fig = plt.figure()
            plt.plot(xs, ys, label=f"AUC={auc:.3f}")
            plt.plot([0,1],[0,1],'--')
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
            plt.legend(loc="lower right"); fig.savefig(path)
        else:
            tp, fp, tn, fn = confusion_counts(y_true, y_pred)
            mat = np.array([[tn, fp],[fn, tp]])
            fig = plt.figure()
            plt.imshow(mat, interpolation="nearest")
            for (i,j), val in np.ndenumerate(mat): plt.text(j, i, str(val), ha="center", va="center")
            plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
            plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True"); fig.savefig(path)
        QtWidgets.QMessageBox.information(self, "Saved Graph", f"Saved: {os.path.abspath(path)}")

    def save(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (self.name_edit.text().strip() or "Group").replace(' ', '_')
        csv_path = os.path.join(EXPORT_DIR, f"binary_data_{name}_{ts}.csv")
        with open(csv_path,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            for r in range(self.table.rowCount()):
                row = [(self.table.item(r,c).text() if self.table.item(r,c) else "") for c in range(self.table.columnCount())]
                if any(cell.strip() for cell in row): writer.writerow(row)
        json_path = os.path.join(EXPORT_DIR, f"binary_results_{name}_{ts}.json")
        with open(json_path,"w") as f: json.dump(self.last_calc or {}, f, indent=2)
        QtWidgets.QMessageBox.information(self,"Saved", f"Saved:\n{os.path.abspath(csv_path)}\n{os.path.abspath(json_path)}")

    
    def batch_from_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder of CSV templates")
        if not folder:
            return
        summary = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(folder, fname)
            try:
                y_true, y_pred, scores = self._collect_from_csv(path)
                if not y_true:
                    continue
                res = binary_metrics(y_true, y_pred)
                item = {
                    'file': fname,
                    'N': res['N'],
                    'Accuracy': res['Accuracy'],
                    'Sensitivity': res['Sensitivity_TPR'],
                    'Specificity': res['Specificity_TNR'],
                    'PPV': res['PPV'],
                    'NPV': res['NPV'],
                    'F1': res['F1'],
                    'Balanced_Accuracy': res['Balanced_Accuracy'],
                    'Youdens_J': res['Youdens_J'],
                    'MCC': res['MCC']
                }
                if scores and len(set(scores))>1:
                    xs, ys, auc = roc_points_auc(y_true, scores)
                    r, p, ap = pr_points_ap(y_true, scores)
                    item['AUC'] = auc
                    item['AP'] = ap
                else:
                    item['AUC'] = float('nan'); item['AP'] = float('nan')
                summary.append(item)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Batch", f"Failed on {fname}: {e}")
        if not summary:
            QtWidgets.QMessageBox.information(self, "Batch", "No valid CSVs found.")
            return
        self._show_batch_summary(summary, folder)

    def _collect_from_csv(self, path):
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
        y_true, y_pred, scores = [], [], []
        for row in rows:
            if not row:
                continue
            try:
                if header and len(header) >= 3:
                    t = int(float(row[1])); p = int(float(row[2]))
                    s = float(row[3]) if len(row) > 3 and row[3] != '' else None
                else:
                    t = int(float(row[1])); p = int(float(row[2]))
                    s = float(row[3]) if len(row) > 3 and row[3] != '' else None
            except:
                continue
            y_true.append(t); y_pred.append(p); scores.append(s)
        score_vals = [s for s in scores if isinstance(s,(int,float))]
        scores = score_vals if len(score_vals) == len(y_true) else []
        return y_true, y_pred, scores

    def _show_batch_summary(self, summary, folder):
        
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Batch Summary")
        v = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(summary), 12)
        headers = ["file","N","Accuracy","Sensitivity","Specificity","PPV","NPV","F1","Balanced_Accuracy","Youdens_J","MCC","AUC"]
        table.setHorizontalHeaderLabels(headers)
        for i, item in enumerate(summary):
            row = [
                item['file'], item['N'], item['Accuracy'], item['Sensitivity'], item['Specificity'],
                item['PPV'], item['NPV'], item['F1'], item['Balanced_Accuracy'], item['Youdens_J'], item['MCC'], item.get('AUC', float('nan'))
            ]
            for j, val in enumerate(row):
                if isinstance(val, float) and not (np.isnan(val)) and headers[j] not in ("file","N"):
                    txt = f"{val:.4f}"
                else:
                    txt = str(val)
                table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        v.addWidget(table)
        h = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save Summary CSV")
        h.addStretch(1); h.addWidget(save_btn)
        v.addLayout(h)

        def _save():
            ensure_export_dir()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(EXPORT_DIR, f"binary_batch_summary_{ts}.csv")
            with open(out, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(headers)
                for i in range(table.rowCount()):
                    w.writerow([table.item(i, j).text() if table.item(i, j) else '' for j in range(table.columnCount())])
            QtWidgets.QMessageBox.information(dlg, "Saved", f"Saved: {out}")
        save_btn.clicked.connect(_save)

        dlg.resize(1000, 500)
        dlg.exec_()




class FleissTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget(10, 5, self)
        self.table.setHorizontalHeaderLabels(["item_id","rater_1","rater_2","rater_3","rater_4"])
        self.result = QtWidgets.QPlainTextEdit(self); self.result.setReadOnly(True)
        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.graph_btn = QtWidgets.QPushButton("Graph")
        self.save_graph_btn = QtWidgets.QPushButton("Save Graph")
        self.save_btn = QtWidgets.QPushButton("Save Data/JSON")
        self.export_results_btn = QtWidgets.QPushButton("Export Results (.txt)")
        self.import_btn = QtWidgets.QPushButton("Import CSV")
        self.add_one_item_btn_f = QtWidgets.QPushButton("Add 1 item")
        self.rem_one_item_btn_f = QtWidgets.QPushButton("Remove 1 item")
        self.add_row_btn = QtWidgets.QPushButton("Add 10 rows")
        self.load_tpl = QtWidgets.QPushButton("Load template")
        self.add_rater_btn_f = QtWidgets.QPushButton("Add Rater")
        self.rem_rater_btn_f = QtWidgets.QPushButton("Remove Rater")
        btn_row = QtWidgets.QHBoxLayout()
        for b in [self.calc_btn, self.graph_btn, self.save_graph_btn, self.save_btn, self.export_results_btn,
                  self.import_btn, self.add_one_item_btn_f, self.rem_one_item_btn_f, self.add_row_btn,
                  self.load_tpl, self.add_rater_btn_f, self.rem_rater_btn_f]:
            btn_row.addWidget(b)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table); layout.addLayout(btn_row); layout.addWidget(self.result)
        self.calc_btn.clicked.connect(self.calculate)
        self.graph_btn.clicked.connect(self.graph)
        self.save_graph_btn.clicked.connect(self.save_graph)
        self.save_btn.clicked.connect(self.save)
        self.export_results_btn.clicked.connect(self.export_results)
        self.import_btn.clicked.connect(self.import_csv)
        self.add_one_item_btn_f.clicked.connect(self.add_one_item)
        self.rem_one_item_btn_f.clicked.connect(self.rem_one_item)
        self.add_row_btn.clicked.connect(self.add_rows)
        self.load_tpl.clicked.connect(self.load_template)
        self.add_rater_btn_f.clicked.connect(self.add_rater)
        self.rem_rater_btn_f.clicked.connect(self.remove_rater)
        self.last = None

    def load_template(self):
        path = os.path.join("examples","fleiss_template.csv")
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self,"Template","Template not found.")
            return
        self._load_csv_to_table(path)

    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_csv_to_table(path)

    def _load_csv_to_table(self, path):
        try:
            with open(path,"r", newline='') as f:
                reader = csv.reader(f); header = next(reader, None); rows = list(reader)
            if header:
                self.table.setColumnCount(len(header))
                self.table.setHorizontalHeaderLabels(header)
            self.table.setRowCount(max(10, len(rows)))
            for i,row in enumerate(rows):
                for j,val in enumerate(row):
                    self.table.setItem(i,j,QtWidgets.QTableWidgetItem(val))
            QtWidgets.QMessageBox.information(self, "Import", f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))

    def add_rows(self):
        self.table.setRowCount(self.table.rowCount()+10)

    def add_rater(self):
        cols = self.table.columnCount()
        self.table.insertColumn(cols)
        self.table.setHorizontalHeaderItem(cols, QtWidgets.QTableWidgetItem(f"rater_{cols}"))

    def remove_rater(self):
        if self.table.columnCount() <= 2:
            QtWidgets.QMessageBox.information(self, "Raters", "At least one rater is required.")
            return
        self.table.removeColumn(self.table.columnCount()-1)

    def add_one_item(self):
        self.table.setRowCount(self.table.rowCount()+1)

    def rem_one_item(self):
        if self.table.rowCount() > 1:
            self.table.setRowCount(self.table.rowCount()-1)

    def _collect(self):
        rows = []
        for r in range(self.table.rowCount()):
            vals = []
            valid = False
            for c in range(1, self.table.columnCount()):
                item = self.table.item(r,c)
                if item and item.text().strip()!="":
                    valid = True
                    vals.append(item.text().strip())
                else:
                    vals.append(None)
            if valid: rows.append(vals)
        return rows

    def calculate(self):
        matrix = self._collect()
        if not matrix:
            QtWidgets.QMessageBox.warning(self,"Input","No valid rows.")
            return
        kappa, P_i, marg = fleiss_kappa_from_raw(matrix)
        low, high = bootstrap_fleiss_ci(matrix, B=500)
        lines = []
        def pct(x):
            return "nan" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{x*100:.2f}%"
        lines.append(f"Fleiss' κ: {'nan' if math.isnan(kappa) else f'{kappa:.4f}'}")
        lines.append(f"Bootstrap 95% CI: [{'nan' if math.isnan(low) else f'{low:.4f}'}, {'nan' if math.isnan(high) else f'{high:.4f}'}]")
        lines.append("")
        P_clean = [x for x in P_i if not math.isnan(x)]
        if P_clean: lines.append(f"Mean per-item agreement (P̄): {np.mean(P_clean):.4f}")
        lines.append("Category prevalence:")
        for cat, p in marg.items(): lines.append(f"  {cat}: {pct(p)}")
        self.result.setPlainText("\n".join(lines))
        self.last = {"kappa": kappa, "ci": (low, high), "per_item": P_i, "marginals": marg, "matrix": matrix}

    def graph(self):
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        marg = self.last["marginals"]
        if not marg:
            QtWidgets.QMessageBox.warning(self,"Graph","No marginals to plot.")
            return
        cats = list(marg.keys()); vals = [marg[c] for c in cats]
        fig = plt.figure(); plt.bar(cats, vals); plt.ylim(0,1); plt.ylabel("Proportion"); plt.title("Category Prevalence")
        self._show_fig(fig)

    def save_graph(self):
        ensure_export_dir()
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORT_DIR, f"fleiss_graph_{ts}.png")
        marg = self.last.get("marginals", {})
        if not marg:
            QtWidgets.QMessageBox.warning(self, "Graph", "No marginals to plot.")
            return
        cats = list(marg.keys()); vals = [marg[c] for c in cats]
        fig = plt.figure(); plt.bar(cats, vals); plt.ylim(0,1); plt.ylabel("Proportion"); plt.title("Category Prevalence")
        fig.savefig(path); QtWidgets.QMessageBox.information(self, "Saved Graph", f"Saved: {os.path.abspath(path)}")

    def _show_fig(self, fig):
        if not hasattr(self, '_fig_windows'):
            self._fig_windows = []
        w = FigureWindow(fig)
        self._fig_windows.append(w)
        w.show()

    def save(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(EXPORT_DIR, f"fleiss_data_{ts}.csv")
        with open(csv_path,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            for r in range(self.table.rowCount()):
                row = [(self.table.item(r,c).text() if self.table.item(r,c) else "") for c in range(self.table.columnCount())]
                if any(cell.strip() for cell in row): writer.writerow(row)
        json_path = os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.json")
        with open(json_path,"w") as f: json.dump(self.last or {}, f, indent=2)
        QtWidgets.QMessageBox.information(self,"Saved", f"Saved:\n{os.path.abspath(csv_path)}\n{os.path.abspath(json_path)}")

    def export_results(self):
        ensure_export_dir()
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"agreement_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"icc_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"binary_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception as _e:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(txt_path)}")




class ICCTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget(10, 4, self)
        self.table.setHorizontalHeaderLabels(["subject_id","rater_1","rater_2","rater_3"])
        self.result = QtWidgets.QPlainTextEdit(self); self.result.setReadOnly(True)
        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.graph_btn = QtWidgets.QPushButton("Graph")
        self.save_graph_btn = QtWidgets.QPushButton("Save Graph")
        self.save_btn = QtWidgets.QPushButton("Save Data/JSON")
        self.export_results_btn = QtWidgets.QPushButton("Export Results (.txt)")
        self.import_btn = QtWidgets.QPushButton("Import CSV")
        self.add_one_item_btn_i = QtWidgets.QPushButton("Add 1 item")
        self.rem_one_item_btn_i = QtWidgets.QPushButton("Remove 1 item")
        self.add_row_btn = QtWidgets.QPushButton("Add 10 rows")
        self.load_tpl = QtWidgets.QPushButton("Load template")
        self.add_rater_btn_i = QtWidgets.QPushButton("Add Rater")
        self.rem_rater_btn_i = QtWidgets.QPushButton("Remove Rater")
        btn_row = QtWidgets.QHBoxLayout()
        for b in [self.calc_btn, self.graph_btn, self.save_graph_btn, self.save_btn, self.export_results_btn,
                  self.import_btn, self.add_one_item_btn_i, self.rem_one_item_btn_i, self.add_row_btn, self.load_tpl,
                  self.add_rater_btn_i, self.rem_rater_btn_i]:
            btn_row.addWidget(b)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table); layout.addLayout(btn_row); layout.addWidget(self.result)
        self.calc_btn.clicked.connect(self.calculate)
        self.graph_btn.clicked.connect(self.graph)
        self.save_graph_btn.clicked.connect(self.save_graph)
        self.save_btn.clicked.connect(self.save)
        self.export_results_btn.clicked.connect(self.export_results)
        self.import_btn.clicked.connect(self.import_csv)
        self.add_one_item_btn_i.clicked.connect(self.add_one_item)
        self.rem_one_item_btn_i.clicked.connect(self.rem_one_item)
        self.add_row_btn.clicked.connect(self.add_rows)
        self.load_tpl.clicked.connect(self.load_template)
        self.add_rater_btn_i.clicked.connect(self.add_rater)
        self.rem_rater_btn_i.clicked.connect(self.remove_rater)
        self.last = None

    def load_template(self):
        path = os.path.join("examples","icc_template.csv")
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self,"Template","Template not found.")
            return
        self._load_csv_to_table(path)

    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_csv_to_table(path)

    def _load_csv_to_table(self, path):
        try:
            with open(path, "r", newline='') as f:
                reader = csv.reader(f); header = next(reader, None); rows = list(reader)
            if header:
                self.table.setColumnCount(len(header))
                self.table.setHorizontalHeaderLabels(header)
            self.table.setRowCount(max(10, len(rows)))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem(val))
            QtWidgets.QMessageBox.information(self, "Import", f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))

    def add_rows(self):
        self.table.setRowCount(self.table.rowCount()+10)

    def add_rater(self):
        cols = self.table.columnCount()
        self.table.insertColumn(cols)
        self.table.setHorizontalHeaderItem(cols, QtWidgets.QTableWidgetItem(f"rater_{cols}"))

    def remove_rater(self):
        if self.table.columnCount() <= 2:
            QtWidgets.QMessageBox.information(self, "Raters", "At least one rater is required.")
            return
        self.table.removeColumn(self.table.columnCount()-1)

    def add_one_item(self):
        self.table.setRowCount(self.table.rowCount()+1)

    def rem_one_item(self):
        if self.table.rowCount() > 1:
            self.table.setRowCount(self.table.rowCount()-1)

    def _collect(self):
        rows = []
        for r in range(self.table.rowCount()):
            rowvals = []
            valid = False
            for c in range(1, self.table.columnCount()):
                item = self.table.item(r,c)
                if item and item.text().strip()!="":
                    valid = True
                    try: rowvals.append(float(item.text().strip()))
                    except: rowvals.append(np.nan)
                else:
                    rowvals.append(np.nan)
            if valid: rows.append(rowvals)
        return np.array(rows, dtype=float)

    def calculate(self):
        X = self._collect()
        if X.size == 0:
            QtWidgets.QMessageBox.warning(self,"Input","No valid rows.")
            return
        icc, comps = icc2_1(X)
        low, high = bootstrap_icc_ci(X, B=1000)
        k = X.shape[1]
        icc2k_point = icc_avg(icc, k, reduce='mean')
        dist = bootstrap_icc_distribution(X, B=1000)
        if isinstance(dist, np.ndarray) and dist.size > 0:
            icc2k_boot = icc_avg(dist, k, reduce=None)
            with np.errstate(all='ignore'):
                low2k, high2k = np.nanpercentile(icc2k_boot, [2.5, 97.5])
            low2k = float(low2k) if np.isfinite(low2k) else float("nan")
            high2k = float(high2k) if np.isfinite(high2k) else float("nan")
        else:
            low2k, high2k = float("nan"), float("nan")

        lines = []
        lines.append(f"ICC(2,1): {'nan' if (icc is None or (isinstance(icc,float) and math.isnan(icc))) else f'{icc:.4f}'}")
        lines.append(f"Bootstrap 95% CI: [{'nan' if (low is None or (isinstance(low,float) and math.isnan(low))) else f'{low:.4f}'}, "
                     f"{'nan' if (high is None or (isinstance(high,float) and math.isnan(high))) else f'{high:.4f}'}]")
        lines.append(f"ICC(2,{k}): {'nan' if (icc2k_point is None or (isinstance(icc2k_point,float) and math.isnan(icc2k_point))) else f'{icc2k_point:.4f}'}  (average of {k} raters)")
        lines.append(f"Bootstrap 95% CI (avg): [{'nan' if math.isnan(low2k) else f'{low2k:.4f}'}, "
                     f"{'nan' if math.isnan(high2k) else f'{high2k:.4f}'}]")
        lines.append("")
        lines.append("Variance components / MS terms:")
        for key in ["MSR","MSC","MSE","SSR","SSC","SSE","n_subjects","k_raters","grand_mean"]:
            v = comps.get(key, None)
            lines.append(f"  {key}: {v:.4f}" if isinstance(v,(int,float)) else f"  {key}: {v}")
        self.result.setPlainText("\n".join(lines))
        self.last = {"icc2_1": icc, "ci2_1": (low,high), "icc2_k": icc2k_point, "ci2_k": (low2k, high2k), "components": comps, "matrix": X.tolist()}

    def graph(self):
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        comps = self.last["components"]
        fig = plt.figure()
        vals = [comps["MSR"], comps["MSC"], comps["MSE"]]
        labels = ["MSR (rows)","MSC (cols)","MSE (error)"]
        plt.bar(labels, vals); plt.title("Variance Components (Mean Squares)"); plt.ylabel("Value")
        self._show_fig(fig)

    def save_graph(self):
        ensure_export_dir()
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORT_DIR, f"icc_graph_{ts}.png")
        comps = self.last["components"]
        fig = plt.figure()
        vals = [comps["MSR"], comps["MSC"], comps["MSE"]]
        labels = ["MSR (rows)","MSC (cols)","MSE (error)"]
        plt.bar(labels, vals); plt.title("Variance Components (Mean Squares)"); plt.ylabel("Value")
        fig.savefig(path); QtWidgets.QMessageBox.information(self, "Saved Graph", f"Saved: {os.path.abspath(path)}")

    def _show_fig(self, fig):
        if not hasattr(self, '_fig_windows'):
            self._fig_windows = []
        w = FigureWindow(fig)
        self._fig_windows.append(w)
        w.show()

    def save(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(EXPORT_DIR, f"icc_data_{ts}.csv")
        with open(csv_path,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            for r in range(self.table.rowCount()):
                row = [(self.table.item(r,c).text() if self.table.item(r,c) else "") for c in range(self.table.columnCount())]
                if any(cell.strip() for cell in row): writer.writerow(row)
        json_path = os.path.join(EXPORT_DIR, f"icc_results_{ts}.json")
        with open(json_path,"w") as f: json.dump(self.last or {}, f, indent=2)
        QtWidgets.QMessageBox.information(self,"Saved", f"Saved:\n{os.path.abspath(csv_path)}\n{os.path.abspath(json_path)}")

    def export_results(self):
        ensure_export_dir()
        # Auto-calc if empty
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"agreement_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        # Auto-calc if empty
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"icc_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        # Auto-calc if empty
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        # Auto-calc if empty
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"binary_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        # Auto-calc if empty
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception as _e:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(EXPORT_DIR, f"icc_results_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(txt_path)}")




class AgreementTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Cohen's κ (2 raters)", "Krippendorff's α (nominal)"])

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)

        self.table = QtWidgets.QTableWidget(10, 3, self)
        self.table.setHorizontalHeaderLabels(["unit_id","rater_1","rater_2"])  # can add more for α
        self.result = QtWidgets.QPlainTextEdit(self); self.result.setReadOnly(True)

        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.graph_btn = QtWidgets.QPushButton("Graph")
        self.save_graph_btn = QtWidgets.QPushButton("Save Graph")
        self.save_btn = QtWidgets.QPushButton("Save Data/JSON")
        self.export_results_btn = QtWidgets.QPushButton("Export Results (.txt)")
        self.import_btn = QtWidgets.QPushButton("Import CSV")
        self.load_tpl = QtWidgets.QPushButton("Load template")
        self.add_rater_btn = QtWidgets.QPushButton("Add Rater")
        self.rem_rater_btn = QtWidgets.QPushButton("Remove Rater")
        self.add_row_btn = QtWidgets.QPushButton("Add 10 rows")
        self.add_one_btn = QtWidgets.QPushButton("Add 1 item")
        self.rem_one_btn = QtWidgets.QPushButton("Remove 1 item")

        btn_row = QtWidgets.QHBoxLayout()
        for b in [self.calc_btn, self.graph_btn, self.save_graph_btn, self.save_btn, self.export_results_btn,
                  self.import_btn, self.load_tpl, self.add_rater_btn, self.rem_rater_btn, self.add_row_btn, self.add_one_btn, self.rem_one_btn]:
            btn_row.addWidget(b)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(mode_row)
        layout.addWidget(self.table)
        layout.addLayout(btn_row)
        layout.addWidget(self.result)

        self.calc_btn.clicked.connect(self.calculate)
        self.graph_btn.clicked.connect(self.graph)
        self.save_graph_btn.clicked.connect(self.save_graph)
        self.save_btn.clicked.connect(self.save)
        self.export_results_btn.clicked.connect(self.export_results)
        self.import_btn.clicked.connect(self.import_csv)
        self.load_tpl.clicked.connect(self.load_template)
        self.add_rater_btn.clicked.connect(self.add_rater)
        self.rem_rater_btn.clicked.connect(self.remove_rater)
        self.add_row_btn.clicked.connect(self.add_rows)
        self.add_one_btn.clicked.connect(self.add_one)
        self.rem_one_btn.clicked.connect(self.rem_one)
        self.last = None

    def add_rows(self):
        self.table.setRowCount(self.table.rowCount()+10)
    def add_one(self):
        self.table.setRowCount(self.table.rowCount()+1)
    def rem_one(self):
        if self.table.rowCount()>1: self.table.setRowCount(self.table.rowCount()-1)

    def add_rater(self):
        cols = self.table.columnCount()
        self.table.insertColumn(cols)
        self.table.setHorizontalHeaderItem(cols, QtWidgets.QTableWidgetItem(f"rater_{cols}"))

    def remove_rater(self):
        if self.table.columnCount() <= 2:
            QtWidgets.QMessageBox.information(self, "Raters", "Need at least two rater columns.")
            return
        self.table.removeColumn(self.table.columnCount()-1)

    def load_template(self):
        
        path = os.path.join("examples","agreement_template.csv")
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "Template", "Template not found.")
            return
        self._load_csv_to_table(path)

    def import_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            self._load_csv_to_table(path)

    def _load_csv_to_table(self, path):
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)
            if header:
                self.table.setColumnCount(len(header))
                self.table.setHorizontalHeaderLabels(header)
            self.table.setRowCount(max(10, len(rows)))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem(val))
            QtWidgets.QMessageBox.information(self, "Import", f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))

    def _collect(self):
        
        data = []
        for r in range(self.table.rowCount()):
            vals = []
            valid = False
            for c in range(1, self.table.columnCount()):
                item = self.table.item(r, c)
                if item and item.text().strip() != "":
                    valid = True
                    vals.append(item.text().strip())
                else:
                    vals.append(None)
            if valid:
                data.append(vals)
        return data

    def calculate(self):
        data = self._collect()
        if not data:
            QtWidgets.QMessageBox.warning(self, "Input", "No valid rows.")
            return
        mode = self.mode_combo.currentText()
        lines = []
        if "Cohen" in mode:
            
            if len(data[0]) != 2:
                QtWidgets.QMessageBox.warning(self, "Cohen's κ", "Exactly 2 rater columns required.")
                return
            r1 = [row[0] for row in data]
            r2 = [row[1] for row in data]
            kappa, info = cohens_kappa(r1, r2)
            lines.append(f"Cohen's κ: {'nan' if (isinstance(kappa,float) and math.isnan(kappa)) else f'{kappa:.4f}'}")
            lines.append(f"Observed agreement (Po): {'nan' if (isinstance(info['po'],float) and math.isnan(info['po'])) else f'{info['po']:.4f}'}")
            lines.append(f"Expected agreement (Pe): {'nan' if (isinstance(info['pe'],float) and math.isnan(info['pe'])) else f'{info['pe']:.4f}'}")
            lines.append(f"N: {info['n']}")
            lines.append("")
            lines.append("Category frequencies:")
            for k, v in info['categories'].items():
                lines.append(f"  {k}: {v}")
            self.last = {"mode": "cohen", "kappa": kappa, "info": info}
        else:
            alpha, info = krippendorff_alpha_nominal(data)
            lines.append(f"Krippendorff's α (nominal): {'nan' if (isinstance(alpha,float) and math.isnan(alpha)) else f'{alpha:.4f}'}")
            lines.append(f"Do (observed disagreement): {'nan' if (isinstance(info['Do'],float) and math.isnan(info['Do'])) else f'{info['Do']:.4f}'}")
            lines.append(f"De (expected disagreement): {'nan' if (isinstance(info['De'],float) and math.isnan(info['De'])) else f'{info['De']:.4f}'}")
            lines.append(f"Total pairs considered: {info['total_pairs']}")
            lines.append("")
            lines.append("Category frequencies:")
            for k, v in sorted(info['category_frequency'].items()):
                lines.append(f"  {k}: {v}")
            self.last = {"mode": "krippendorff", "alpha": alpha, "info": info}
        self.result.setPlainText("\n".join(lines))

    def graph(self):
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        
        info = self.last['info']
        cats = []
        vals = []
        if self.last['mode'] == 'cohen':
            freq = info['categories']
        else:
            freq = info['category_frequency']
        if not freq:
            QtWidgets.QMessageBox.information(self, "Graph", "No categories to plot.")
            return
        for k, v in sorted(freq.items()):
            cats.append(str(k))
            vals.append(v)
        s = sum(vals)
        props = [v/s if s else 0 for v in vals]
        fig = plt.figure()
        plt.bar(cats, props)
        plt.ylim(0, 1)
        plt.ylabel("Proportion")
        plt.title("Category Prevalence")
        self._show_fig(fig)

    def _show_fig(self, fig):
        if not hasattr(self, '_fig_windows'):
            self._fig_windows = []
        w = FigureWindow(fig)
        self._fig_windows.append(w)
        w.show()

    def save_graph(self):
        ensure_export_dir()
        if self.last is None:
            self.calculate()
        if self.last is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = 'cohen' if self.last['mode'] == 'cohen' else 'krippendorff'
        path = os.path.join(EXPORT_DIR, f"agreement_{fname}_graph_{ts}.png")
        info = self.last['info']
        freq = info['categories'] if self.last['mode'] == 'cohen' else info['category_frequency']
        cats, vals = zip(*sorted(freq.items())) if freq else ([], [])
        s = sum(vals) if vals else 0
        props = [v/s if s else 0 for v in vals]
        fig = plt.figure(); plt.bar(cats, props); plt.ylim(0,1); plt.ylabel("Proportion"); plt.title("Category Prevalence")
        fig.savefig(path); QtWidgets.QMessageBox.information(self, "Saved Graph", f"Saved: {os.path.abspath(path)}")

    def save(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(EXPORT_DIR, f"agreement_data_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            for r in range(self.table.rowCount()):
                row = [(self.table.item(r, c).text() if self.table.item(r, c) else "") for c in range(self.table.columnCount())]
                if any(cell.strip() for cell in row):
                    w.writerow(row)
        json_path = os.path.join(EXPORT_DIR, f"agreement_results_{ts}.json")
        with open(json_path, 'w') as f:
            json.dump(self.last or {}, f, indent=2)
        QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{csv_path}\n{json_path}")

    def export_results(self):
        ensure_export_dir()
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"agreement_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"icc_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"fleiss_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.abspath(os.path.join(EXPORT_DIR, f"binary_results_{ts}.txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results As", default, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(path)}")
        
        if not self.result.toPlainText().strip():
            try:
                self.calculate()
            except Exception as _e:
                pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(EXPORT_DIR, f"agreement_results_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Export Results", f"Saved: {os.path.abspath(txt_path)}")




class FigureWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        super().__init__()
        self.setWindowTitle("Chart")
        canvas = FigureCanvas(fig)
        self.setCentralWidget(canvas)




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, user):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.user = user
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        
        topbar = QtWidgets.QHBoxLayout()
        topbar.addWidget(QtWidgets.QLabel(f"Signed in as: {user}"))
        topbar.addStretch(1)
        topbar.addWidget(QtWidgets.QLabel("Date:"))
        self.date_edit = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        topbar.addWidget(self.date_edit)

        self.group_combo = QtWidgets.QComboBox(); self.group_combo.addItem("Group-1")
        self.add_group_btn = QtWidgets.QPushButton("Add Group")
        self.rem_group_btn = QtWidgets.QPushButton("Remove Group")
        self.rename_group_btn = QtWidgets.QPushButton("Rename Group")
        topbar.addStretch(1)
        topbar.addWidget(QtWidgets.QLabel("Group:"))
        topbar.addWidget(self.group_combo)
        topbar.addWidget(self.add_group_btn); topbar.addWidget(self.rem_group_btn); topbar.addWidget(self.rename_group_btn)
        layout.addLayout(topbar)

        
        self.tabs = QtWidgets.QTabWidget()
        self.binary = BinaryTab(); self.fleiss = FleissTab(); self.icc = ICCTab(); self.agree = AgreementTab()
        self.tabs.addTab(self.binary, "Binary Classification")
        self.tabs.addTab(self.fleiss, "Fleiss' κ (Agreement)")
        self.tabs.addTab(self.icc, "ICC (Reliability)")
        self.tabs.addTab(self.agree, "Cohen's κ & Krippendorff's α")
        layout.addWidget(self.tabs)

        
        self.group_states = {"Group-1": self._snapshot()}
        self.group_combo.currentTextChanged.connect(self._on_group_change)
        self.add_group_btn.clicked.connect(self._add_group)
        self.rem_group_btn.clicked.connect(self._remove_group)
        self.rename_group_btn.clicked.connect(self._rename_group)

        
        footer = QtWidgets.QHBoxLayout()
        self.prev_sessions_combo = QtWidgets.QComboBox()
        self.load_session_btn = QtWidgets.QPushButton("Load Session")
        self.save_all_btn = QtWidgets.QPushButton("Save Session")
        footer.addWidget(QtWidgets.QLabel("Previous Sessions:"))
        footer.addWidget(self.prev_sessions_combo)
        footer.addWidget(self.load_session_btn)
        footer.addStretch(1)
        footer.addWidget(self.save_all_btn)
        layout.addLayout(footer)
        self.save_all_btn.clicked.connect(self._save_all_groups)
        self.load_session_btn.clicked.connect(self._load_selected_session)

        ensure_export_dir()
        self._refresh_sessions_list()

    def _snapshot(self):
        def table_to_list(table):
            rows = []
            for r in range(table.rowCount()):
                row = []
                row_has = False
                for c in range(table.columnCount()):
                    item = table.item(r,c); txt = "" if item is None else item.text()
                    if txt.strip()!="": row_has = True
                    row.append(txt)
                if row_has: rows.append(row)
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            return {"headers": headers, "rows": rows}
        
        return {
            "binary": {**table_to_list(self.binary.table), "label": self.binary.name_edit.text()},
            "fleiss": table_to_list(self.fleiss.table),
            "icc": table_to_list(self.icc.table),
            "agree": table_to_list(self.agree.table)
        }

    def _restore(self, state):
        def fill_table(table, data):
            headers = data.get("headers", []); rows = data.get("rows", [])
            if headers: table.setColumnCount(len(headers)); table.setHorizontalHeaderLabels(headers)
            table.setRowCount(max(10, len(rows)))
            for i,row in enumerate(rows):
                for j,val in enumerate(row):
                    table.setItem(i,j,QtWidgets.QTableWidgetItem(val))
        b = state.get("binary", {})
        fill_table(self.binary.table, b)
        if "label" in b: self.binary.name_edit.setText(b.get("label", ""))
        fill_table(self.fleiss.table, state.get("fleiss", {}))
        fill_table(self.icc.table, state.get("icc", {}))
        fill_table(self.agree.table, state.get("agree", {}))

    def _on_group_change(self, name):
        prev = getattr(self, "current_group", None)
        if prev is not None:
            self.group_states[prev] = self._snapshot()
        self.current_group = name
        st = self.group_states.get(name, None)
        if st is None:
            st = self._snapshot(); self.group_states[name] = st
        self._restore(st)

    def _add_group(self):
        n = self.group_combo.count() + 1
        name = f"Group-{n}"
        self.group_states[name] = self._snapshot()
        self.group_combo.addItem(name); self.group_combo.setCurrentText(name)

    def _remove_group(self):
        if self.group_combo.count() == 1:
            QtWidgets.QMessageBox.information(self,"Groups","At least one group is required.")
            return
        name = self.group_combo.currentText()
        self.group_states.pop(name, None)
        idx = self.group_combo.currentIndex()
        self.group_combo.removeItem(idx)

    def _rename_group(self):
        old = self.group_combo.currentText()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename Group", "New name:", text=old)
        if not ok or not new.strip():
            return
        new = new.strip()
        if new in self.group_states and new != old:
            QtWidgets.QMessageBox.warning(self, "Rename", "A group with that name already exists.")
            return
        
        self.group_states[old] = self._snapshot()
        self.group_states[new] = self.group_states.pop(old)
        
        idx = self.group_combo.currentIndex()
        self.group_combo.setItemText(idx, new)

    def _refresh_sessions_list(self):
        ensure_export_dir()
        files = sorted([f for f in os.listdir(EXPORT_DIR) if f.startswith("emtas4_all_") and f.endswith(".json")])
        self.prev_sessions_combo.clear(); self.prev_sessions_combo.addItems(files)

    def _load_selected_session(self):
        name = self.prev_sessions_combo.currentText()
        if not name:
            QtWidgets.QMessageBox.information(self, "Load", "No session selected.")
            return
        path = os.path.join(EXPORT_DIR, name)
        try:
            with open(path, "r") as f: payload = json.load(f)
            self.group_states = payload.get("groups", {})
            self.group_combo.clear()
            for g in self.group_states.keys(): self.group_combo.addItem(g)
            if self.group_combo.count()>0:
                self.group_combo.setCurrentIndex(0)
                self._on_group_change(self.group_combo.currentText())
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _save_all_groups(self):
        ensure_export_dir()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORT_DIR, f"emtas4_all_{ts}.json")
        if self.group_combo.currentText():
            self.group_states[self.group_combo.currentText()] = self._snapshot()
        payload = {"user": self.user, "date": self.date_edit.date().toString(QtCore.Qt.ISODate), "groups": self.group_states}
        with open(path, "w") as f: json.dump(payload, f, indent=2)
        self._refresh_sessions_list()
        QtWidgets.QMessageBox.information(self,"Saved", f"Saved: {path}")




def main():
    try:
        import numpy, matplotlib
    except Exception as e:
        print('[ERROR] Dependency missing:', e)
        raise
    app = QtWidgets.QApplication([])
    login = LoginDialog()
    if login.exec_() != QtWidgets.QDialog.Accepted: return
    user = login.get_user()
    win = MainWindow(user); win.resize(1250, 800); win.show()
    app.exec_()


if __name__ == "__main__":
    main()
