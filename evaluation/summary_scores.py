# Get results of evaluation
#
# Prints the accuracy scoreboard for one merged results.jsonl, plus — for negation
# tags — WHY the failures happened: the four exclusive failure causes,
# how systematic the failures are, and the paired negated-vs-positive contrast.
# Everything is also written to the summary CSV.

import argparse
import os
import sys

import numpy as np
import pandas as pd

# `python evaluation/summary_scores.py` puts evaluation/ on sys.path, NOT the repo
# root, so neg_ext is not importable without help.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from neg_ext import analysis as A


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--csv", type=str, default=None,
                    help="Where to write the summary CSV "
                         "(default: <filename>_summary.csv next to the results).")
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

df = pd.read_json(args.filename, orient="records", lines=True)

# Measure overall success

n_images = len(df)
n_prompts = len(df.groupby('metadata'))
pct_images = df['correct'].mean()
prompt_correct = df.groupby('metadata')['correct'].any()
pct_prompts = prompt_correct.mean()

print("Summary")
print("=======")
print(f"Total images: {n_images}")
print(f"Total prompts: {n_prompts}")
print(f"% correct images: {pct_images:.2%}")
print(f"% correct prompts: {pct_prompts:.2%}")
print()

# By group

task_scores = []
rows = []  # tidy rows for the CSV

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    acc = task_df['correct'].mean()
    task_scores.append(acc)
    correct, total = int(task_df['correct'].sum()), len(task_df)
    lo, hi = A.wilson_interval(correct, total)
    print(f"{tag:<16} = {acc:.2%} ({correct} / {total})  95% CI [{lo:.2%}, {hi:.2%}]")
    rows.append({"type": "task", "name": tag, "correct": correct,
                 "total": total, "accuracy": acc,
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
print()

overall = float(np.mean(task_scores)) if task_scores else float("nan")
print(f"Overall score (avg. over tasks): {overall:.5f}")

# ── Why the failures happened (negation tags only) ────────────────────────────
#
# Row-level `correct` cannot say which slot broke, so the statements are reconstructed
# from each row's metadata and matched against the evaluator's own reason wording.
# Negated and positive violations are not tie-broken against each other.

res = A.result_rows_from_jsonl(args.filename)
if not res.empty:
    slots = A.explode_result_slots(res)
    causes = A.failure_causes(res, slots)
    shares = A.failure_cause_shares(causes, group=("tag",))

    print()
    print("Failure causes (share of FAILING rows, mutually exclusive)")
    print("=========================================================")
    if shares.empty:
        print("no failing rows")
    else:
        present = [c for c in A.CAUSE_ORDER
                   if c in set(shares["cause"].astype(str))]
        piv = (shares.pivot_table(index="tag", columns="cause", values="share",
                                  observed=True, fill_value=0.0)
               .reindex(columns=present, fill_value=0.0))
        n_fail = causes.groupby("tag").size()
        head = "".join(f"{c:>18}" for c in present)
        print(f"{'tag':<20}{'failing':>8}{head}")
        for tag, r in piv.iterrows():
            cells = "".join(f"{r[c]:>17.1%} " for c in present)
            print(f"{str(tag):<20}{int(n_fail.get(tag, 0)):>8}{cells}")
        print()
        print("  'negated only' is a lower bound on negation-specific failure;")
        print("  'both violated' rows are NOT charged to either polarity.")

        for _, r in shares.iterrows():
            n_tag_fail = int(n_fail.get(r["tag"], 0))
            rows.append({"type": "failure_cause",
                         "name": f"{r['tag']}/{r['cause']}",
                         "correct": int(r["rows"]), "total": n_tag_fail,
                         "accuracy": float(r["share"]), "ci_lo": "", "ci_hi": ""})

    # Systematic vs stochastic: a prompt at 0-of-n is a model that cannot do the
    # task; one at n/2 is a coin flip. Neither pct_images nor pct_prompts shows this.
    counts = A.prompt_success_counts(res)
    if not counts.empty:
        n_max = int(counts["n"].max())
        zero = float((counts["k"] == 0).mean())
        allc = float((counts["k"] == counts["n"]).mean())
        print()
        print("Systematicity")
        print("=============")
        print(f"prompts failing ALL {n_max} samples : {zero:.1%}")
        print(f"prompts passing ALL {n_max} samples : {allc:.1%}")
        print(f"prompts in between (stochastic)    : {1 - zero - allc:.1%}")
        rows.append({"type": "systematicity", "name": "all_samples_failed",
                     "correct": int((counts["k"] == 0).sum()), "total": len(counts),
                     "accuracy": round(zero, 4), "ci_lo": "", "ci_hi": ""})
        rows.append({"type": "systematicity", "name": "all_samples_passed",
                     "correct": int((counts["k"] == counts["n"]).sum()),
                     "total": len(counts), "accuracy": round(allc, 4),
                     "ci_lo": "", "ci_hi": ""})

    # Paired negated-vs-positive, same image. Empty unless the setting has both
    # polarities (pos<P>_neg<N> with P>0 and N>0), so pos0_neg* prints nothing.
    paired = A.paired_polarity(slots, group=("category",))
    if not paired.empty:
        print()
        print("Negated vs positive, paired within the same image")
        print("================================================")
        print(f"{'category':<22}{'neg':>8}{'pos':>8}{'delta':>9}{'p':>12}")
        for _, r in paired.iterrows():
            print(f"{r['category']:<22}{r['neg_rate']:>8.1%}{r['pos_rate']:>8.1%}"
                  f"{r['delta']:>+9.1%}{r['p']:>12.2e}")
            rows.append({"type": "paired_polarity", "name": r["category"],
                         "correct": "", "total": int(r["rows"]),
                         "accuracy": float(r["delta"]), "ci_lo": "", "ci_hi": ""})
        print("  violation rate of negated vs positive statements; delta>0 = negation harder")

# ── Write the summary as a CSV (in addition to the prints) ────────────────────

rows += [
    {"type": "summary", "name": "images", "correct": int(df['correct'].sum()),
     "total": n_images, "accuracy": pct_images},
    {"type": "summary", "name": "prompts", "correct": int(prompt_correct.sum()),
     "total": n_prompts, "accuracy": pct_prompts},
    {"type": "summary", "name": "overall_avg_over_tasks", "correct": "",
     "total": "", "accuracy": overall},
]

csv_path = args.csv or (os.path.splitext(args.filename)[0] + "_summary.csv")
if os.path.dirname(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
pd.DataFrame(rows, columns=["type", "name", "correct", "total", "accuracy",
                            "ci_lo", "ci_hi"]).to_csv(csv_path, index=False)
print(f"\n[summary] wrote CSV: {csv_path}")
