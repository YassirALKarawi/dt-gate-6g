# scripts/make_figs.py
import argparse, os, glob
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def _safe(v: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in v)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["data/outputs/epochs_log.csv"],
                   help="ملفات CSV (تقبل wildcards)")
    p.add_argument("--outdir", default="data/outputs")
    p.add_argument("--roc-bins", type=int, default=200, help="عدد نقاط الرسم على ROC")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # وسّع الـglobs إن وُجدت
    files = []
    for pat in args.inputs:
        hit = glob.glob(pat)
        files.extend(hit if hit else [pat])

    # حمّل كل الموجود فقط
    dfs = []
    missing = []
    for f in files:
        if os.path.isfile(f):
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"[warn] تعذّر قراءة {f}: {e}")
        else:
            missing.append(f)
    if not dfs:
        raise SystemExit(f"لا توجد ملفات صالحة للقراءة: {files}")

    df = pd.concat(dfs, ignore_index=True)

    # تأكد من الأعمدة الأساسية
    needed = {"t","L99","risk","variant"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"الأعمدة المطلوبة {needed} غير مكتملة. الأعمدة المتاحة: {list(df.columns)}")

    # ملخّص سريع لكل variant
    summary_rows = []
    for v in df["variant"].dropna().unique():
        sub = df[df["variant"] == v].sort_values("t")
        tag = _safe(str(v))

        # L99 vs epoch
        plt.figure()
        plt.plot(sub["t"].values, sub["L99"].values)
        plt.title(f"L99 — {v}")
        plt.xlabel("Epoch"); plt.ylabel("ms")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"l99_{tag}.png"))
        plt.close()

        # Risk vs epoch
        plt.figure()
        plt.plot(sub["t"].values, sub["risk"].values)
        plt.title(f"Risk — {v}")
        plt.xlabel("Epoch"); plt.ylabel("Risk")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"risk_{tag}.png"))
        plt.close()

        # GateAccept (rolling mean 20)
        if "accepted" in sub.columns:
            r = sub["accepted"].rolling(20, min_periods=1).mean()
            plt.figure()
            plt.plot(sub["t"].values, r.values)
            plt.title(f"Gate Accept (rolling) — {v}")
            plt.xlabel("Epoch"); plt.ylabel("Accept rate")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"gate_accept_{tag}.png"))
            plt.close()

        # ROC (لو وُجد label_atk)
        if {"label_atk", "risk"}.issubset(sub.columns):
            try:
                # نستخدم (1 - risk) كـ score أعلى = هجومي أقل
                y_true = sub["label_atk"].astype(int).values
                y_score = (1.0 - sub["risk"].values).astype(float)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                plt.plot([0,1],[0,1], linestyle="--")
                plt.title(f"ROC — {v}")
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"roc_{tag}.png"))
                plt.close()

                summary_rows.append({
                    "variant": v,
                    "AUC": roc_auc,
                    "L99_mean": float(np.mean(sub["L99"])),
                    "GateAccept_%": float(100.0 * sub.get("accepted", pd.Series([0]*len(sub))).mean()),
                    "n_rows": int(len(sub))
                })
            except Exception as e:
                print(f"[warn] ROC فشل لـ {v}: {e}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.outdir, "figs_summary.csv")
        summary.to_csv(summary_path, index=False)

    print("Saved figures to", args.outdir)

if __name__ == "__main__":
    main()
