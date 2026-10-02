# NegT2IBench

NegT2IBench is a benchmark that tests whether text-to-image models follow **negation**: prompts that deny an attribute (*"a photo of a non-black handbag"*) or a spatial relation (*"the cup is not left of the bowl"*).

Scoring uses detectors and geometry, not an LLM judge:

- **RF-DETR-XL** checks that every requested object is present (and counts instances).
- **SigLIP2** classifies colour and material.
- **SAM 2** masks plus geometry decide directional, proximity and comparative relations.
- **Depth-Anything-V2** decides front/behind.

An image counts as correct only if *every* statement in its prompt holds, positive and negated alike.

The repository contains:

| folder | what |
|---|---|
| `negt2ibench_prompts/` | the benchmark: 8 settings × 6 categories × 100 prompts = **4,800 prompts** |
| `generation/` | one CLI for the 10 benchmarked text-to-image models (`generation/models/registry.py`) |
| `evaluation/` | the evaluator, per-image scoring, summaries and the report |
| `human_validation/` | agreement of NegT2IBench and 10 other metrics with human labels |
| `prompt_generation/`, `neg_ext/` | code that built the prompts (only needed for a new prompt set) |
| `scripts/` | stage scripts for a local run |
| `jobs/` | the same stages as SLURM jobs |
| `config.sh` | every path and environment setting |
| `CONTEXT.md` | glossary of the benchmark's terms |

---

## 1. Benchmark layout

`negt2ibench_prompts/<setting>/<category>.jsonl` holds one prompt per line.

**Settings.** `pos<P>_neg<N>` means P positive and N negated statements per prompt. There are eight: `pos1_neg0 pos0_neg1 pos2_neg0 pos1_neg1 pos0_neg2 pos2_neg1 pos1_neg2 pos2_neg2`.

**Categories:**

| category | example prompt (from `pos1_neg1`) |
|---|---|
| `neg_attr_color` | a photo of a red pizza and a non-yellow cell phone |
| `neg_attr_material` | a photo of a wooden pizza and a non-metal cell phone |
| `neg_rel_directional` | … where the scissors is above the donut and is not above the boat |
| `neg_rel_proximity` | … where the scissors is near the donut and is not near the stop sign |
| `neg_rel_comparative` | … where the scissors is shorter than the boat and is not bigger than the donut |
| `neg_rel_depth` | … where the scissors is in front of the donut and is not in front of the stop sign |

Each row also carries the machine-readable statements (`slots`, `include`) that the evaluator checks.

---

## 2. Setup

Requirements:

- Linux, an NVIDIA GPU, CUDA 12, and `conda` or `micromamba`.
- Tested on NVIDIA H100/H200-class GPUs. Evaluation and the default generator (`zimage`) fit in 32 GB of GPU memory; `flux2` and `flux2-4bit` need more.

```bash
git clone <this repo> NegT2IBench && cd NegT2IBench

# 1. (optional) edit config.sh: CACHE_DIR, OUTPUT_ROOT, ENV_MANAGER, ENV_NAME
# 2. create the env + the RF-DETR overlay, and verify both
bash scripts/0_setup_env.sh
# 3. check the install (46 checks; --skip-heavy runs the 44 that need no GPU models)
source config.sh && negt2i_activate eval && python sanity_check.py
```

Evaluation runs in two layers:

- **Base env (`requirements.txt`):** torch 2.6. Used for prompts, generation, summaries and reports.
- **RF-DETR overlay (`requirements-rfdetr.txt`):** RF-DETR needs torch 2.14. `0_setup_env.sh` installs it with `pip install --target envs/rfdetr_overlay`, and `negt2i_activate eval` puts it in front of the base env, for evaluation only. These are the exact versions the benchmark was run with.

**Model weights** download on first use into `CACHE_DIR` (default `./cache`): Hugging Face to `cache/huggingface`, RF-DETR to `cache/rfdetr`. Running every generator needs several hundred GB, so point `CACHE_DIR` at a large disk. Some models need extra setup:

- **`hidream-4bit`:** needs access to the gated `meta-llama/Meta-Llama-3.1-8B-Instruct`. Run `huggingface-cli login` first.
- **`negationclip`:** needs the NegationCLIP ViT-B/32 weights at `CLIP_WEIGHTS` (default `cache/negationclip_ViT-B32.pth`).
- **`blip3o-next`:** needs its own env and a checkout of the `BLIP3o-NEXT` branch of `JiuhaiChen/BLIP3o` at `BLIP3O_NEXT_ROOT` (default `external/BLIP3o-NEXT`). Install that repo's requirements into a separate env, then run with `ENV_NAME=<that env>`.

---

## 3. Configuration: `config.sh`

Every script sources `config.sh`. Every value has the form `VAR="${VAR:-default}"`, so you can either edit the file or override per run (`MODEL=sd35 bash scripts/...`). Print the resolved values with:

```bash
NEGT2I_VERBOSE=1 source config.sh
```

| variable | default | meaning |
|---|---|---|
| `PROMPT_DIR` | `negt2ibench_prompts` | prompts to generate from |
| `OUTPUT_ROOT` | `outputs` | everything the pipeline writes |
| `DATA_ROOT` | `outputs/images` | generated images + per-image verdicts |
| `RESULTS_SUBDIR` | `results` | verdict folder name inside each setting |
| `REPORT_DIR` | `outputs/reports` | stage-5 reports |
| `HUMAN_EVAL_DIR` | `outputs/human_validation` | metric scores + human comparison |
| `LOG_DIR` | `outputs/logs` | SLURM / local logs |
| `CACHE_DIR` | `cache` | `HF_HOME`, `TORCH_HOME`, `RF_HOME`, `CLIP_DOWNLOAD_ROOT` |
| `ENV_MANAGER`, `ENV_NAME` | `auto`, `negt2ibench` | `conda` / `micromamba` / `none` (use the active python) |
| `RFDETR_OVERLAY` | `envs/rfdetr_overlay` | RF-DETR install, evaluation only |
| `MODEL`, `N_SAMPLES`, `SEED` | `zimage`, `4`, `42` | generation |

**Evaluator parameters.** The evaluator has no runtime options: every parameter is a constant in `evaluation/` and is written to each run's `results/_config.json`.

| component | model / parameter | value |
|---|---|---|
| detection | RF-DETR-XL (`RFDETRSegXLarge`) | presence threshold 0.03; counted instance ≥ 0.5; exactly one instance per object (`skis` exempt) |
| colour / material | SigLIP2 `ViT-SO400M-14-SigLIP2-378` (`webli`) on the detector-mask crop | confusion buffer 0.5 logits (colour, `gray` allowed as an answer), 0 (material) |
| relation masks | SAM 2.1 `facebook/sam2.1-hiera-small` | closing radius 5 px, components ≥ 5 % of the largest |
| directional | mask centroids | margin 0.075, axis 0.45 |
| proximity | mask gap vs mean radius | near below 0.75 × radius, else far |
| comparative | mask area / vertical extent | ratio > 1.05 |
| depth | Depth-Anything-V2-Small on facing surfaces | window 16 px; confidence 0.05 + 0.10 × gap/R |

---

## 4. Running the benchmark locally

```bash
# quick smoke test: 1 prompt x 1 image per category of one setting (~5 min)
MODELS=zimage SETTINGS=pos1_neg1 LIMIT=1 N_SAMPLES=1 bash scripts/run_all.sh

# full benchmark for one or more models (4,800 prompts x 4 images each)
MODELS="sd35 flux2" bash scripts/run_all.sh
```

Or stage by stage:

```bash
MODEL=sd35 bash scripts/2_generate_images.sh   # GPU: images
MODEL=sd35 bash scripts/3_evaluate_images.sh   # GPU: per-image verdicts
MODEL=sd35 bash scripts/4_summarize.sh         # CPU: scores per setting + overall
bash scripts/5_report.sh                       # CPU: report across all evaluated models
```

Every stage accepts these filters and knobs:

- `SETTINGS="pos1_neg1 pos0_neg2"` and `CATEGORIES="neg_attr_color neg_rel_depth"`
- `LIMIT=N`: only the first N prompts per file (stage 2)
- `SKIP_DONE=1`: skip categories already fully scored (stage 3)
- `FIGURES=report|none|core|all`: which figures to draw (stage 5)

Stage 2 keeps existing images, so an interrupted run resumes where it stopped.

**Text-to-image models** (`MODEL=`; steps/guidance defaults from each model card, override with `GEN_ARGS="--steps 20 --scale 5"`):

| key | model | HF id |
|---|---|---|
| `sd35` | Stable Diffusion 3.5 Large | `stabilityai/stable-diffusion-3.5-large` |
| `flux2` | FLUX.2 klein 9B | `black-forest-labs/FLUX.2-klein-9B` |
| `flux2-4bit` | FLUX.2 dev, bnb 4-bit | `diffusers/FLUX.2-dev-bnb-4bit` |
| `qwen` | Qwen-Image | `Qwen/Qwen-Image-2512` |
| `zimage` | Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` |
| `hidream-4bit` | HiDream-I1 Full, NF4 | `HiDream-ai/HiDream-I1-Full` |
| `janus-pro` | Janus-Pro 7B | `deepseek-community/Janus-Pro-7B` |
| `blip3o-next` | BLIP3o-NEXT SFT 3B | `BLIP3o/BLIP3o-NEXT-SFT-3B` |
| `hunyuanimage-distilled` | HunyuanImage-2.1 distilled, NF4 | `hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers` |
| `negationclip` | SD 1.5 + NegationCLIP guidance | `runwayml/stable-diffusion-v1-5` |

To add a model, register a loader in `generation/models/registry.py`.

### Output paths

```
outputs/
  images/<model>/<setting>/<category>/<idx>/metadata.jsonl      prompt row
  images/<model>/<setting>/<category>/<idx>/samples/0000N.png   generated images
  images/<model>/<setting>/results/<category>.jsonl             one verdict per image
  images/<model>/<setting>/results/_config.json                 evaluator options + environment
  images/<model>/<setting>/results/all_results_summary.csv      scores for this setting
  images/<model>/results/all_results_summary.csv                scores over all settings
  reports/main/README.md                                        report index (start here)
  reports/main/report__comparison.md                            all settings side by side
  reports/main/report__<setting>.md, *.csv, failure_report.png
  human_validation/comparison.txt                               metric-vs-human agreement
  logs/
```

**Verdict rows.** Each row of `results/<category>.jsonl` has:

- `correct` (bool) and `reason`: which statement failed and why.
- `filename` and `prompt`.
- The raw detections, so a verdict can be audited.

**Accuracy.** Two numbers are reported:

- *Image accuracy* (primary): the share of images where every statement holds.
- *Prompt accuracy*: the share of prompts with at least one correct image.

---

## 5. Running on SLURM

The jobs run one (setting, category) pair per array task: 48 tasks for the full benchmark. They call the same stage scripts and read the same `config.sh`. Add any site-specific `sbatch` flags in `SBATCH_ARGS` or on the `sbatch` command line. Submit from the repository root.

```bash
# everything, with dependencies: generate -> evaluate (per model) -> report
MODELS="sd35 flux2" bash jobs/submit_all.sh

# a subset (the array is sized automatically)
MODELS=sd35 SETTINGS="pos1_neg1 pos0_neg1" bash jobs/submit_all.sh
MODELS=sd35 SKIP_GENERATE=1 bash jobs/submit_all.sh          # re-score existing images

# or submit single stages yourself
MODEL=sd35 sbatch jobs/generate.sbatch                       # --array=0-47
MODEL=sd35 sbatch jobs/evaluate.sbatch
sbatch jobs/report.sbatch
sbatch jobs/human_validation.sbatch
```

Logs go to `outputs/logs/<job>_<jobid>_<task>.out`.

---

## 6. Human validation

`human_validation/` measures how well NegT2IBench, and 10 commonly used metrics, agree with human judgements on the same generated images.

**Human annotations.** `HUMAN_ANNOTATIONS` (default `human_annotations/annotations.jsonl`) has one JSON row per (annotator, image):

```json
{"datapointId": "flux2-pos1_neg0-neg_rel_proximity-00016-samples-00002",
 "annotator": "annotator_07", "model": "flux2", "posNeg": "pos1_neg0",
 "tag": "neg_rel_proximity", "category": "proximity", "prompt": "...",
 "objectResponses":    [{"object": "cup", "response": true}, ...],
 "conditionResponses": [{"conditionId": "cond_rel_0", "response": false}]}
```

`datapointId` is `<model>-<setting>-<category>-<idx>-samples-<sample>`, i.e. the image `DATA_ROOT/<model>/<setting>/<category>/<idx>/samples/<sample>.png`. Before sharing a raw export, strip personal data (emails, nicknames, ids):

```bash
python -m human_validation.anonymize_annotations raw_export.jsonl human_annotations/annotations.jsonl
```

**Run.** First run stages 2–3 for the annotated models, so `DATA_ROOT` holds their images and verdicts. Then:

```bash
# NegT2IBench vs humans (base env, CPU):
bash scripts/human_validation/run_human_validation.sh
```

This writes:

- `outputs/human_validation/input.jsonl`: the scored manifest.
- `neggenbench_both.jsonl` and `human_scores.jsonl`.
- `comparison.txt`, with three evaluations:
  1. binary accuracy and Cohen's κ vs the human label;
  2. Spearman/Kendall/pairwise accuracy vs the human score;
  3. same-prompt ranking accuracy.

  Each comes with a leave-one-annotator-out human ceiling and breakdowns by model, category and setting.

**Other metrics.** The comparison metrics need their own envs, because their pinned dependencies conflict:

```bash
bash human_validation/envs/create_environment.sh               # negt2i-clip-metrics: CLIPScore, BLIPScore,
                                                               #   NegCLIP, ImageReward, PickScore, HPSv2
bash human_validation/envs/create_vqa_environment.sh           # negt2i-vqa-metrics: Qwen2.5-VL, InternVL3
bash human_validation/envs/create_remaining_vqa_environments.sh  # envs/negt2i-vqascore, envs/negt2i-mplug
```

`create_remaining_vqa_environments.sh` needs a patched checkout of `t2v_metrics` at `external/t2v_metrics-clip-flant5-xxl` (see the comment in the script). mPLUG also needs the official mPLUG repo at `MPLUG_ROOT`.

Then name the metrics and the python of each env:

```bash
CLIP_METRICS="clipscore blipscore negclip imagereward pickscore hpsv2" \
CLIP_PYTHON=$(micromamba run -n negt2i-clip-metrics which python) \
VQA_METRICS="vqascore qwen25vl internvl mplug" \
VQA_PYTHON=$(micromamba run -n negt2i-vqa-metrics which python) \
VQASCORE_PYTHON=envs/negt2i-vqascore/bin/python MPLUG_PYTHON=envs/negt2i-mplug/bin/python \
MPLUG_ROOT=external/mPLUG \
bash scripts/human_validation/run_human_validation.sh
```

Scoring resumes by default (`OVERWRITE=1` rescores). The VQA judges are scored both on the whole caption and per element, with the same questions the annotators saw.

**Single metric:**

```bash
python -m human_validation.score     input.jsonl out.jsonl --metric clipscore --image-root outputs/images
python -m human_validation.score_vqa input.jsonl out.jsonl --metric qwen25vl --image-root outputs/images --mode both
```

**Unannotated full sweep.** To score a metric over every generated image:

```bash
python -m human_validation.build_geneval_manifest --output outputs/human_validation/full_sweep/manifest.jsonl
```

Then run the scorers on that manifest; `python -m human_validation.build_neggenbench_full` emits NegT2IBench's own verdict for the same rows.

---

## 7. Building a new prompt set (optional)

The released prompts are in `negt2ibench_prompts/`; this step is only needed to regenerate or extend them. It writes to `outputs/prompts/` and refuses to overwrite the released set.

The generators use `Qwen/Qwen3-30B-A3B-Instruct-2507` (4-bit, ~32 GB) to choose plausible object combinations. `LLM_ARGS="--no-llm"` runs rule-based only, on CPU.

```bash
NUM_POS=1 NUM_NEG=1 NUM_PROMPTS=100 bash scripts/1_create_prompts.sh
PROMPT_DIR=outputs/prompts MODEL=sd35 bash scripts/2_generate_images.sh
```

---

## Acknowledgements and license

The evaluation layout and the original object vocabulary build on [GenEval](https://github.com/djghosh13/geneval) (Ghosh et al., NeurIPS 2023). MIT license (`LICENSE`).
