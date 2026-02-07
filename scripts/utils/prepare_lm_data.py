# scripts/utils/prepare_lm_data.py
import os, json, argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext_out", default="data_cache/wikitext2_raw")
    ap.add_argument("--c4_calib_out", default="data_cache/c4_en_calib.jsonl")
    ap.add_argument("--calib_n", type=int, default=2048)
    ap.add_argument("--streaming", action="store_true", help="use streaming for C4")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.c4_calib_out), exist_ok=True)

    # 1) cache wikitext2 raw
    if not os.path.exists(args.wikitext_out):
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        wt.save_to_disk(args.wikitext_out)
        print(f"[OK] saved wikitext2 raw to: {args.wikitext_out}")
    else:
        print(f"[SKIP] wikitext2 raw exists: {args.wikitext_out}")

    # 2) build C4 calib jsonl
    if not os.path.exists(args.c4_calib_out):
        ds = load_dataset("allenai/c4", "en", split="train", streaming=args.streaming)
        n = 0
        with open(args.c4_calib_out, "w", encoding="utf-8") as f:
            for ex in ds:
                text = ex.get("text", "")
                if not text or len(text) < 30:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
                if n >= args.calib_n:
                    break
        print(f"[OK] saved C4 calib jsonl: {args.c4_calib_out}  (n={n})")
    else:
        print(f"[SKIP] C4 calib exists: {args.c4_calib_out}")

if __name__ == "__main__":
    main()
