"""Phase 0 task 0d: measure embedding throughput on the A2000 and verify each
candidate model's licence and parameter count from its live model card.

The model choice is made on measured recall-per-GPU-hour, not on recollection,
so this script reports the two numbers that decide it: texts/sec on realistic
inputs, and the extrapolated wall-clock for a full 20M-record pass.
"""
import argparse, csv, time
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source", required=True, help="a source TSV to draw realistic benchmark text from")
ap.add_argument("--models", nargs="+",
                default=["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base"])
args = ap.parse_args()
CANDIDATES = args.models
N_BENCH = 20000
TOTAL_RECORDS = 20_000_000  # train + test S2/S3 combined, worst case

# Realistic inputs: actual test rows, "name | address" as the blocking key.
texts = []
with open(args.source, newline="", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); next(r)
    for row in r:
        texts.append(f"{row[1]} | {row[2]}")
        if len(texts) >= N_BENCH: break
print(f"benchmark corpus: {len(texts)} real test rows, "
      f"mean {sum(len(t) for t in texts)/len(texts):.0f} chars\n")

api = HfApi()
for name in CANDIDATES:
    print(f"=== {name} ===")
    info = api.model_info(name)
    cfg = AutoConfig.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    # Licence straight off the model card metadata, not from memory.
    lic = (info.card_data.get("license") if info.card_data else None) or "UNKNOWN"
    print(f"  licence (model card): {lic}")
    print(f"  parameters (counted): {n_params/1e6:.1f}M   hidden dim: {cfg.hidden_size}")

    for bs in (256, 512):
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
                enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
                out = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                (out * mask).sum(1) / mask.sum(1)   # mean pooling
                n += len(batch)
        torch.cuda.synchronize()
        rate = n / (time.time() - t0)
        hrs = TOTAL_RECORDS / rate / 3600
        print(f"  batch={bs:4d}: {rate:8.0f} texts/sec  ->  {hrs:5.2f} h for {TOTAL_RECORDS/1e6:.0f}M records")
    del model; torch.cuda.empty_cache()
    print()
