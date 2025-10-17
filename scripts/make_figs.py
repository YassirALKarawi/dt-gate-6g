import argparse, glob, os, pandas as pd, matplotlib.pyplot as plt
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["data/outputs/epochs_log.csv"])
    p.add_argument("--outdir", default="data/outputs")
    args=p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # simple plots by variant
    df=pd.concat([pd.read_csv(f) for f in args.inputs], ignore_index=True)
    for v in df['variant'].unique():
        sub=df[df['variant']==v]
        plt.figure(); plt.plot(sub['t'], sub['L99']); plt.title(f"L99 — {v}"); plt.xlabel("Epoch"); plt.ylabel("ms")
        plt.savefig(os.path.join(args.outdir, f"l99_{v.replace(' ','_')}.png")); plt.close()
        plt.figure(); plt.plot(sub['t'], sub['risk']); plt.title(f"Risk — {v}"); plt.xlabel("Epoch"); plt.ylabel("Risk")
        plt.savefig(os.path.join(args.outdir, f"risk_{v.replace(' ','_')}.png")); plt.close()
    print("Saved figures to", args.outdir)
if __name__ == "__main__":
    main()
