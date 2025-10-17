def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=700)
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--jsonl", type=str, default=None)   # جديد: مصدر تليمتري ملفّي
    args=p.parse_args()
    df = run_all(variant=args.variant, epochs=args.epochs, seeds=args.seeds, jsonl=args.jsonl)
    import os
    os.makedirs("data/outputs", exist_ok=True)
    df.to_csv("data/outputs/epochs_log.csv", index=False)
    save_summary_and_csv(df)
    print("Logs saved to data/outputs/epochs_log.csv")
