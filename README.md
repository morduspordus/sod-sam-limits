# What limits the refinement of SOD with SAM?

Code for the paper *What Limits the Refinement of Salient Object Detection with the Segment Anything
Model?* (O. Veksler). It refines the saliency map of a trained salient object detection (SOD) model with
SAM 2.1, by a graph cut with six parameters or by a small learned fusion head, and contains every
evaluation and analysis of the paper: the baselines, the ablations, the post-processing variants, the
oracles, the error decomposition, and the multi-annotator evaluation.

## Layout

- `sodsam/` the library
  - `data.py` datasets and the on-disk layout of cached outputs
  - `sam_prompting.py` SAM 2.1 prompts, including the zoom-in mask prompt of the paper (`from_zoom`)
  - `fusion.py`, `fusion2.py` the graph-cut fusion: unary and pairwise terms, oriented repulsive terms,
    exact minimization with PyMaxflow (the paper's model is `fusion2` with `unary="linear"`,
    `pairwise="paper"`)
  - `metrics.py` F^max, MAE, S-measure and E-measure, following PySODMetrics
  - `ideas.py`, `labelcost.py`, `unary_learn.py`, `ranker.py`, ... the post-processing variants
- `scripts/` command-line entry points; every script starts with a docstring that explains its use
- `tests/` checks of the energy, its exact minimization, and the metrics (`python -m pytest tests`)

## Setup

Python 3.12 with the packages in `requirements.txt`, plus:

- [SAM 2](https://github.com/facebookresearch/sam2) with the `sam2.1_hiera_large` checkpoint, in
  `third_party/sam2` or wherever `SAM2_ROOT` points;
- the SOD models' official code and weights (SelfReformer in `third_party/SelfReformer` or
  `SELFREFORMER_ROOT`), or their released saliency maps, imported with `scripts/run_sod.py --import-dir`;
- the datasets under `data/saliency` (or `SALIENCY_DATA`), in the layout of `sodsam/data.py`.

Cached outputs go to `output/` (or `SODSAM_OUT`) and results to `results/`.

## Pipeline

    python scripts/run_sod.py --model selfreformer                       # soft SOD maps
    python scripts/run_sam.py --sod-model selfreformer --prompt mask_zoom   # SAM 2.1 masks, zoom-in prompt
    python scripts/tune2.py --sod-model selfreformer --prompt mask_zoom \
        --datasets MSOD MSRA-B1K MSRA-B4K --unary linear --pairwise paper --output binary \
        --tune-half all --n-random 300 --rounds 2 --n-refine 80          # the six graph-cut parameters
    python scripts/fuse_eval.py --sod-model selfreformer --prompt mask_zoom \
        --methods sod sam_conf "v2:results/tune2/<result>.json"           # evaluation on the benchmarks
    python scripts/train_head.py --sod-model selfreformer --prompt mask_zoom \
        --train-dataset MSRA-B4K --tag selfreformer_hz                    # the learned fusion head

The analyses: `oracle.py` and `pixel_oracle.py` (oracles), `granularity.py` (the per-image choice of one
whole candidate within the pixel oracle), `review_eval.py` (untuned rules, boundary measures, the gain by
error type), `error_types.py` (remaining errors), `error_robust.py` (the error decomposition under other
weightings and thresholds, and at the pixel level), `multi_answer_eval.py` and `annotator_support.py`
(multi-annotator data), `bootstrap_ci.py` (confidence intervals), `paper_tables.py` and
`review_tables.py` (the tables of the paper and the supplementary material, generated from the result
files). Other encodings of the mask prompt: `run_sam.py --prompt mask_zoom_binary mask_zoom_logit
mask_zoom_signed mask_zoom_pts`, then `tune2.py` and `fuse_eval.py` as above with the same prompt
(tuned on MSOD and MSRA-B1K).

Added in the revision (v1.1):

- `oracle_auto.py` per-image oracles that also choose among SAM's automatic masks
  (`run_sam_extra.py --prompt auto` or `auto_loose`), and greedy unions of them;
- `fixed_f.py` F at the fixed threshold 0.5 and adaptive F, with paired bootstrap intervals;
- `val_overlap.py` the overlap of the fusion head's training set (MSRA-B4K, or `--train DUTS-TR`)
  with its validation images; to retrain the head without the overlapping images, set
  `SODSAM_EXCLUDED=sodsam/excluded_ids_dedup.json` before `train_head.py`;
- `sodc_infer.py` saliency maps of MDSAM, VSCode-S and VST++-B from their official code and released
  weights (`SODC_ROOT`, `SODC_DATA`), and `cmp_maps.py`, which compares them with the released maps.

Added in the second round of the revision:

- `oracle_auto_types.py` the gain of the automatic-mask oracle split by the error type of the SOD map;
- `gain_type_ci.py` bootstrap intervals of the share of the fusion's gain on images with a boundary error;
- `sliver_stats.py` the small cells of constant signature of the pixel-level oracle;
- `suppl_stats_tables.py` and `multi_stats_table.py` the supplementary tables of the oracle statistics, the
  per-type loss reduction, the two analyses above, and the multi-annotator statistics;
- the fusion head with the plain prompt, as the zoom-in head but with `--prompt mask_soft --seed <s>
  --tag <sod_model>_hsoft_s<s>` (seeds 0 to 2); seed 0 is the "plain prompt" row of the head ablations.

## License

MIT; see `LICENSE`. SAM 2, the SOD models and the datasets have their own licenses.
