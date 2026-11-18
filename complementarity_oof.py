"""Likelihood-ratio complementarity test on the out-of-fold radiomic score, recomputed from the cross-validation checkpoints."""
import config_cox as cfg

import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    if not (cfg.CKPT_DIR / "cv_risk_main.csv").exists():
        sys.exit("checkpoints_cox\\cv_risk_main.csv not found: run the pipeline first.")

    from checks_cox import oof_scores
    from data_cox import load_dataset
    from evaluation_cox import likelihood_ratio_rad_score

    ds = load_dataset()
    space = next((sp for sp in cfg.FEATURE_SPACES if sp.startswith("radiomic")),
                 None)
    if space is None:
        sys.exit("No radiomic space in config_cox.FEATURE_SPACES.")

    score = oof_scores(ds, space)
    if score is None or (score != score).any():
        sys.exit(f"Missing out-of-fold scores for some patients in {space}: "
                 f"rerun the cv stage.")

    lr = likelihood_ratio_rad_score(ds, score)
    result = {"score_source": f"out-of-fold ({space}, mean over CV repeats)",
              "n_patients": int(len(ds.df)), "n_events": int(ds.event.sum()),
              **{k: v for k, v in lr.items() if k != "model_combined"}}

    print(f"Radiomic space: {space}  (n = {result['n_patients']}, "
          f"events = {result['n_events']})")
    print(f"  C-index clinical only  : {lr['c_clinical']:.3f}")
    print(f"  C-index rad-score only : {lr['c_rad_score']:.3f}")
    print(f"  C-index combined       : {lr['c_combined']:.3f}")
    print(f"  rad-score adds to clinical: chi2(1) = {lr['lr_score_added']:.2f}, "
          f"p = {lr['p_score_added']:.4g}")
    print(f"  clinical adds to rad-score: chi2({len(ds.clinical_features)}) = "
          f"{lr['lr_clinical_added']:.2f}, p = {lr['p_clinical_added']:.4g}")
    print("  (C-indices here are apparent values of the plain Cox models on "
          "the out-of-fold score; use the cv results for discrimination.)")

    path = cfg.RESULTS_DIR / "cox_likelihood_ratio_oof.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
