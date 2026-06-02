"""Phase 4 — train the LightGBM suspicion model on Snorkel soft labels.

    uv run python -m rrs.modeling.train                 # full Philadelphia run (~10-15 min)
    uv run python -m rrs.modeling.train --sample 100000  # fast debug subset
    uv run python -m rrs.modeling.train --n-trials 15    # lighter Optuna search
    uv run python -m rrs.modeling.train --no-shap --no-ablation

Pipeline:
  1. Join Phase 3 features to Phase 2 `p_suspicious`, split by time at 2020 (`dataset.py`).
  2. Optuna tunes LightGBM against a *time-based* validation fold — the last pre-2020 year
     (2019) held out of the fit. Objective: validation AUC vs the binarized soft label.
     The model is *trained* on the continuous probability via the `cross_entropy` objective.
  3. Refit the final model on the full pre-2020 period at the tuned settings, evaluate on
     the held-out 2020+ test period (AUC / AP / precision@k / logloss).
  4. Validation beyond metrics: a feature-space synthetic-injection probe, a held-out-
     heuristic ablation (drop the embedding-similarity family, check it's recovered), and a
     top-100 manual-audit table.
  5. Predict `p_fake` for every review and cache a SHAP TreeExplainer.

Outputs:
    models/lgbm_suspicion.txt      — the booster (LightGBM text format)
    models/shap_explainer.pkl      — pickled shap.TreeExplainer for Phase 4/5/6 reuse
    models/predictions.parquet     — review_id → p_fake for all reviews (Phase 5 input)
    models/feature_importance.parquet
    reports/model_<metro>.md       — metrics, importances, validation, top-100 audit
"""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from rrs.config import DEFAULT_METRO, REPORTS_DIR, ROOT
from rrs.modeling.dataset import (
    HARD_LABEL_THRESHOLD,
    LABELS_PATH,
    LEAKY_AGGREGATE_FEATURES,
    Dataset,
    load_joined,
    time_split,
)

MODELS_DIR = ROOT / "models"

# Embedding-derived similarity scalars — the features most directly tied to the
# duplicate-text labeling function. Dropped in the ablation to test recovery.
EMBEDDING_SIM_FEATURES = ("max_sim_to_user_history", "max_sim_to_business_reviews")

# Validation fold boundary: reviews in [TUNE_VAL_START, SPLIT_DATE) are the tuning
# validation set; everything before TUNE_VAL_START is the tuning fit set.
TUNE_VAL_START = datetime(2019, 1, 1)


def _to_float_numpy(X: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Polars frame → float64 numpy with nulls as NaN (LightGBM handles NaN natively)."""
    return X.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def _now() -> float:
    return time.perf_counter()


# --------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------

def _base_params() -> dict:
    return {
        "objective": "cross_entropy",  # native soft-label [0,1] target
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "force_row_wise": True,
        "num_threads": 0,  # all cores
        # Optuna reuses one Dataset across trials while varying min_child_samples; with
        # feature pre-filtering on, lowering it below the first trial's value is fatal.
        "feature_pre_filter": False,
    }


def tune(ds: Dataset, n_trials: int, seed: int = 42) -> tuple[dict, int]:
    """Optuna search against a time-based validation fold. Returns (best_params,
    best_num_boost_round)."""
    import lightgbm as lgb
    import optuna

    dates = ds.train_meta.get_column("date")
    val_mask = (dates >= TUNE_VAL_START).to_numpy()
    fit_mask = ~val_mask

    Xtr = _to_float_numpy(ds.X_train, ds.feature_cols)
    soft = ds.soft_train.to_numpy()
    hard = ds.hard_train.to_numpy()

    X_fit, soft_fit = Xtr[fit_mask], soft[fit_mask]
    X_val, hard_val = Xtr[val_mask], hard[val_mask]
    print(
        f"  tuning fit={fit_mask.sum():,} (pre-2019)  val={val_mask.sum():,} (2019)  "
        f"val positive rate={hard_val.mean():.3f}",
        flush=True,
    )

    # Train Dataset carries the *soft* target (objective reads this); the validation
    # Dataset carries the *hard* label (the AUC metric reads this).
    fit_ds = lgb.Dataset(X_fit, label=soft_fit, feature_name=ds.feature_cols, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=hard_val, reference=fit_ds, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = _base_params() | {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "seed": seed,
        }
        booster = lgb.train(
            params,
            fit_ds,
            num_boost_round=2000,
            valid_sets=[val_ds],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score["valid"]["auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_params = _base_params() | {k: v for k, v in best.params.items()}
    best_params["bagging_freq"] = 1
    best_params["seed"] = seed
    best_iter = int(best.user_attrs["best_iteration"])
    print(f"  best val AUC={best.value:.4f}  num_boost_round={best_iter}", flush=True)
    return best_params, best_iter


def fit_final(ds: Dataset, params: dict, num_boost_round: int, feature_cols: list[str]):
    """Refit on the full pre-2020 period at fixed rounds (no early stopping — all train
    data is used, so we trust the tuned iteration count, scaled up modestly)."""
    import lightgbm as lgb

    X = _to_float_numpy(ds.X_train.select(feature_cols), feature_cols)
    train_ds = lgb.Dataset(X, label=ds.soft_train.to_numpy(), feature_name=feature_cols)
    # The final fit sees ~20% more data than the tuning fit (2019 folded back in), so a
    # few extra rounds is reasonable; clamp to a sane floor.
    rounds = max(50, int(round(num_boost_round * 1.1)))
    booster = lgb.train(params, train_ds, num_boost_round=rounds,
                        callbacks=[lgb.log_evaluation(0)])
    return booster


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def evaluate(booster, ds: Dataset, feature_cols: list[str]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    Xte = _to_float_numpy(ds.X_test.select(feature_cols), feature_cols)
    p = booster.predict(Xte)
    hard = ds.hard_test.to_numpy()
    soft = ds.soft_test.to_numpy()

    order = np.argsort(-p)

    def precision_at(k: int) -> float:
        return float(hard[order[:k]].mean())

    return {
        "n_test": int(len(p)),
        "test_positive_rate": float(hard.mean()),
        "auc": float(roc_auc_score(hard, p)),
        "average_precision": float(average_precision_score(hard, p)),
        "logloss_vs_soft": _soft_logloss(soft, p),
        "pearson_vs_soft": float(np.corrcoef(soft, p)[0, 1]),
        "precision_at_100": precision_at(100),
        "precision_at_1000": precision_at(1000),
        "pred": p,
    }


def _soft_logloss(soft: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy between continuous target `soft` and prediction `p`."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(soft * np.log(p) + (1 - soft) * np.log(1 - p)))


# --------------------------------------------------------------------------------------
# Validation: synthetic injection + ablation
# --------------------------------------------------------------------------------------

def synthetic_injection(booster, ds: Dataset, feature_cols: list[str], n: int = 2000) -> dict:
    """Take real low-suspicion test reviews and perturb their features into a classic
    botting profile (duplicate text, fresh account, no social, in a posting burst,
    extreme rating). Measure how many the model flips to a high `p_fake`.

    This is a feature-space injection — it does not synthesize review text and re-run the
    Phase 3 pipeline (deferred), but it directly tests whether the learned decision
    surface responds to the canonical fake-review signal."""
    hard = ds.hard_test.to_numpy()
    soft = ds.soft_test.to_numpy()
    # Source pool: genuinely benign-looking reviews (low soft label, not already positive).
    benign = np.where((soft < 0.1) & (hard == 0))[0]
    if len(benign) == 0:
        return {"n": 0, "note": "no benign source rows"}
    take = benign[:n]
    X = ds.X_test.select(feature_cols)[take]

    overrides = {
        "is_in_burst_window": 1,
        "max_sim_to_user_history": 0.98,
        "max_sim_to_business_reviews": 0.95,
        "account_age_days_at_review": 2,
        "total_reviews": 1,
        "fan_count": 0,
        "friend_count": 0,
        "photo_count": 0,
        "reviews_per_month": 30.0,
        "stars": 5.0,
        "hours_since_prev_review_on_business": 0.5,
        "business_review_count_at_time": 1,
    }
    exprs = [pl.lit(v).alias(c) for c, v in overrides.items() if c in feature_cols]
    injected = X.with_columns(exprs)

    base_p = booster.predict(_to_float_numpy(X, feature_cols))
    inj_p = booster.predict(_to_float_numpy(injected, feature_cols))
    return {
        "n": int(len(take)),
        "base_mean_p_fake": float(base_p.mean()),
        "injected_mean_p_fake": float(inj_p.mean()),
        "recall_at_0.5": float((inj_p >= 0.5).mean()),
        "recall_at_0.8": float((inj_p >= 0.8).mean()),
        "median_lift": float(np.median(inj_p - base_p)),
    }


def ablation_duplicate_text(ds: Dataset, params: dict, num_boost_round: int) -> dict:
    """Held-out-heuristic test. Retrain dropping the embedding-similarity features (the
    model's most direct line to the duplicate-text signal) and check whether the model
    still ranks duplicate-text-flagged reviews — i.e. recovers the heuristic from other
    features. We compare AUC of each model against the `lf_duplicate_text` positive flag
    on the test set."""
    from sklearn.metrics import roc_auc_score

    # Load the duplicate-text LF vote for the test reviews.
    lf = pl.read_parquet(LABELS_PATH).select(["review_id", "lf_duplicate_text"])
    test_lf = ds.test_meta.join(lf, on="review_id", how="left").get_column("lf_duplicate_text")
    # LF votes: 1=SUSPICIOUS, 0=AUTHENTIC, -1=ABSTAIN. Positive class = flagged suspicious.
    dup_pos = (test_lf == 1).cast(pl.Int8).to_numpy()
    if dup_pos.sum() == 0:
        return {"note": "no duplicate-text positives in test set"}

    full_cols = ds.feature_cols
    ablated_cols = [c for c in full_cols if c not in EMBEDDING_SIM_FEATURES]

    full = fit_final(ds, params, num_boost_round, full_cols)
    ablated = fit_final(ds, params, num_boost_round, ablated_cols)

    p_full = full.predict(_to_float_numpy(ds.X_test.select(full_cols), full_cols))
    p_abl = ablated.predict(_to_float_numpy(ds.X_test.select(ablated_cols), ablated_cols))
    return {
        "dup_text_positives_in_test": int(dup_pos.sum()),
        "auc_full_vs_dup_flag": float(roc_auc_score(dup_pos, p_full)),
        "auc_ablated_vs_dup_flag": float(roc_auc_score(dup_pos, p_abl)),
    }


# --------------------------------------------------------------------------------------
# SHAP + importances
# --------------------------------------------------------------------------------------

def gain_importance(booster, feature_cols: list[str]) -> pl.DataFrame:
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    return (
        pl.DataFrame({"feature": feature_cols, "gain": gains, "splits": splits})
        .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_frac"))
        .sort("gain", descending=True)
    )


def build_shap(booster, ds: Dataset, feature_cols: list[str], sample: int = 5000):
    """Fit a TreeExplainer and compute SHAP on a sample for the global summary. Returns
    (explainer, mean_abs_shap DataFrame). The explainer is pickled for downstream reuse;
    we never persist 1.9M × n_features of SHAP values."""
    import shap

    explainer = shap.TreeExplainer(booster)
    Xtr = ds.X_train.select(feature_cols)
    take = min(sample, Xtr.height)
    Xs = _to_float_numpy(Xtr[:take], feature_cols)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):  # some shap versions return per-class lists
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    summary = (
        pl.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort("mean_abs_shap", descending=True)
    )
    return explainer, summary


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def _ablation_verdict(full_auc: float, abl_auc: float) -> str:
    """Data-driven read of the duplicate-text ablation. Interprets the *absolute* AUCs
    (does the model track the heuristic at all?) and the drop when similarity features
    are removed (does it lean on them?)."""
    if full_auc < 0.6:
        return (
            "Both AUCs sit near chance (0.5): `p_fake` barely tracks the duplicate-text "
            "flag even with the similarity features. The LabelModel weights this heuristic "
            "lightly and the gbdt gives the embedding-similarity features near-zero gain, "
            "so the duplicate-text signal is effectively independent of what the model "
            "learned — it is not reproduced rather than recovered-from-correlates."
        )
    if abl_auc >= full_auc - 0.05:
        return (
            "The ablated AUC stays close to the full AUC, so the model recovers the "
            "duplicate-text signal from correlated features (e.g. brevity, burst, account "
            "age) rather than relying on the embedding-similarity scalars."
        )
    return (
        "Removing the similarity features drops the AUC materially, so the model's line to "
        "the duplicate-text signal runs mainly through those embedding scalars and is only "
        "partially recoverable from other features."
    )


def write_report(
    metrics: dict,
    gain: pl.DataFrame,
    shap_summary: pl.DataFrame | None,
    injection: dict,
    ablation: dict,
    top_audit: pl.DataFrame,
    params: dict,
    n_train: int,
    feature_cols: list[str],
    path: Path,
) -> None:
    lines: list[str] = []
    w = lines.append
    w(f"# Suspicion model — {DEFAULT_METRO}\n")
    w("LightGBM trained on Snorkel `p_suspicious` soft labels (`cross_entropy` objective), "
      "time-split at 2020.\n")

    w("## Test metrics (2020+ holdout)\n")
    w("| metric | value |")
    w("|---|---|")
    w(f"| train reviews (pre-2020) | {n_train:,} |")
    w(f"| test reviews (2020+) | {metrics['n_test']:,} |")
    w(f"| test positive rate (p≥{HARD_LABEL_THRESHOLD}) | {metrics['test_positive_rate']:.4f} |")
    w(f"| ROC AUC | {metrics['auc']:.4f} |")
    w(f"| average precision | {metrics['average_precision']:.4f} |")
    w(f"| logloss vs soft target | {metrics['logloss_vs_soft']:.4f} |")
    w(f"| Pearson(pred, soft) | {metrics['pearson_vs_soft']:.4f} |")
    w(f"| precision@100 | {metrics['precision_at_100']:.3f} |")
    w(f"| precision@1000 | {metrics['precision_at_1000']:.3f} |")
    w("")

    w("> **AUC is against the LabelModel's own binarized output**, so it measures how well "
      "the discriminative model reproduces and generalizes the weak labels — not ground "
      "truth. There is no labeled ground truth here; treat `p_fake` as a calibrated second "
      "opinion, never a verdict.\n")

    w("## Feature importance (gain)\n")
    w("| feature | gain frac | splits |")
    w("|---|---|---|")
    for r in gain.head(20).iter_rows(named=True):
        w(f"| {r['feature']} | {r['gain_frac']:.3f} | {r['splits']} |")
    w("")

    if shap_summary is not None:
        w("## SHAP global importance (mean \\|SHAP\\|, 5k-row sample)\n")
        w("| feature | mean abs SHAP |")
        w("|---|---|")
        for r in shap_summary.head(20).iter_rows(named=True):
            w(f"| {r['feature']} | {r['mean_abs_shap']:.4f} |")
        w("")

    w("## Validation — synthetic injection (feature-space)\n")
    if injection.get("n"):
        w(f"Perturbed {injection['n']:,} benign test reviews into a botting feature profile "
          "(duplicate text, fresh account, no social, in burst, 5★).\n")
        w("| metric | value |")
        w("|---|---|")
        w(f"| base mean p_fake | {injection['base_mean_p_fake']:.4f} |")
        w(f"| injected mean p_fake | {injection['injected_mean_p_fake']:.4f} |")
        w(f"| recall @ p_fake≥0.5 | {injection['recall_at_0.5']:.3f} |")
        w(f"| recall @ p_fake≥0.8 | {injection['recall_at_0.8']:.3f} |")
        w(f"| median lift | {injection['median_lift']:.4f} |")
        w("")
    else:
        w(f"_skipped: {injection.get('note', 'n/a')}_\n")

    w("## Validation — held-out heuristic ablation (duplicate text)\n")
    if "auc_full_vs_dup_flag" in ablation:
        full_auc = ablation["auc_full_vs_dup_flag"]
        abl_auc = ablation["auc_ablated_vs_dup_flag"]
        w("Retrained without the embedding-similarity features; checked whether the model "
          "still ranks `lf_duplicate_text`-flagged reviews.\n")
        w("| model | AUC vs duplicate-text flag |")
        w("|---|---|")
        w(f"| full (with similarity features) | {full_auc:.4f} |")
        w(f"| ablated (no similarity features) | {abl_auc:.4f} |")
        w(f"\nDuplicate-text positives in test: {ablation['dup_text_positives_in_test']:,}. "
          + _ablation_verdict(full_auc, abl_auc) + "\n")
    else:
        w(f"_skipped: {ablation.get('note', 'n/a')}_\n")

    w("## Manual audit — top 100 by p_fake (test set)\n")
    w("Eyeball these for precision@100. Columns: rank, p_fake, soft label, stars, "
      "char_length, account age at review, in burst, max sim to user history.\n")
    w("| # | p_fake | p_susp | stars | chars | acct_age_d | burst | sim_user |")
    w("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(top_audit.head(100).iter_rows(named=True), 1):
        w(f"| {i} | {r['p_fake']:.3f} | {r['p_suspicious']:.3f} | {r['stars']:.0f} | "
          f"{r['char_length']} | {r['account_age_days_at_review']} | "
          f"{r['is_in_burst_window']} | {r['max_sim_to_user_history']:.2f} |")
    w("")

    w("## Known limitations\n")
    w("- **Temporal leakage in per-user aggregates.** These features are computed over each "
      "user's full history through the Jan-2022 snapshot, so a pre-2020 training row "
      f"reflects later behavior: {', '.join(LEAKY_AGGREGATE_FEATURES)}. "
      "`account_age_days_snapshot` was dropped in favour of the point-in-time "
      "`account_age_days_at_review`; the rest are retained for signal and flagged here.\n")
    w("- **No ground truth.** Labels are weak (Snorkel over 10 heuristics); every metric is "
      "relative to those heuristics, not verified fake/genuine reviews.\n")
    w(f"- **Model settings.** {len(feature_cols)} features, params: "
      f"`{ {k: params[k] for k in sorted(params) if k not in _base_params()} }`.\n")

    path.write_text("\n".join(lines))
    print(f"  wrote {path}", flush=True)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 — train the suspicion model.")
    ap.add_argument("--n-trials", type=int, default=40, help="Optuna trials")
    ap.add_argument("--sample", type=int, default=None, help="debug: subsample N reviews")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = _now()
    print("→ load + join features/labels", flush=True)
    joined = load_joined()
    if args.sample:
        joined = joined.sort(pl.col("review_id").hash()).head(args.sample)
    ds = time_split(joined)
    print(f"  train={ds.X_train.height:,}  test={ds.X_test.height:,}  "
          f"features={len(ds.feature_cols)}  ({_now() - t0:.1f}s)", flush=True)

    print(f"→ Optuna tuning ({args.n_trials} trials)", flush=True)
    t = _now()
    params, best_iter = tune(ds, n_trials=args.n_trials, seed=args.seed)
    print(f"  ({_now() - t:.1f}s)", flush=True)

    print("→ fit final model (full pre-2020)", flush=True)
    t = _now()
    booster = fit_final(ds, params, best_iter, ds.feature_cols)
    print(f"  ({_now() - t:.1f}s)", flush=True)

    print("→ evaluate on 2020+ test", flush=True)
    metrics = evaluate(booster, ds, ds.feature_cols)
    print(f"  AUC={metrics['auc']:.4f}  AP={metrics['average_precision']:.4f}  "
          f"P@100={metrics['precision_at_100']:.3f}  P@1000={metrics['precision_at_1000']:.3f}",
          flush=True)

    print("→ synthetic injection probe", flush=True)
    injection = synthetic_injection(booster, ds, ds.feature_cols)
    if injection.get("n"):
        print(f"  injected recall@0.5={injection['recall_at_0.5']:.3f}  "
              f"mean p_fake {injection['base_mean_p_fake']:.3f} → "
              f"{injection['injected_mean_p_fake']:.3f}", flush=True)

    ablation: dict = {"note": "skipped (--no-ablation)"}
    if not args.no_ablation:
        print("→ held-out heuristic ablation (duplicate text)", flush=True)
        t = _now()
        ablation = ablation_duplicate_text(ds, params, best_iter)
        if "auc_full_vs_dup_flag" in ablation:
            print(f"  AUC vs dup-flag: full={ablation['auc_full_vs_dup_flag']:.4f}  "
                  f"ablated={ablation['auc_ablated_vs_dup_flag']:.4f}  ({_now() - t:.1f}s)",
                  flush=True)

    gain = gain_importance(booster, ds.feature_cols)
    gain.write_parquet(MODELS_DIR / "feature_importance.parquet")

    shap_summary = None
    if not args.no_shap:
        print("→ SHAP explainer", flush=True)
        t = _now()
        explainer, shap_summary = build_shap(booster, ds, ds.feature_cols)
        with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
            pickle.dump(explainer, f)
        print(f"  cached shap_explainer.pkl  ({_now() - t:.1f}s)", flush=True)

    print("→ predict p_fake for all reviews", flush=True)
    all_X = _to_float_numpy(joined.select(ds.feature_cols), ds.feature_cols)
    p_all = booster.predict(all_X)
    preds = joined.select(["review_id"]).with_columns(
        pl.Series("p_fake", p_all, dtype=pl.Float32)
    )
    preds.write_parquet(MODELS_DIR / "predictions.parquet")
    print(f"  wrote models/predictions.parquet ({preds.height:,} rows)", flush=True)

    # Top-100 audit table from the test set.
    audit = (
        ds.test_meta
        .with_columns(pl.Series("p_fake", metrics["pred"], dtype=pl.Float32))
        .join(joined.select(["review_id", "p_suspicious", "stars", "char_length",
                             "account_age_days_at_review", "is_in_burst_window",
                             "max_sim_to_user_history"]),
              on="review_id", how="left")
        .sort("p_fake", descending=True)
    )

    booster.save_model(str(MODELS_DIR / "lgbm_suspicion.txt"))
    print("  saved models/lgbm_suspicion.txt", flush=True)

    report_path = REPORTS_DIR / f"model_{DEFAULT_METRO}.md"
    write_report(metrics, gain, shap_summary, injection, ablation, audit, params,
                 ds.X_train.height, ds.feature_cols, report_path)

    print(f"\n✓ Phase 4 complete in {(_now() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
