# Shape radiomics and clinical-molecular variables provide complementary, time-dependent prognostic information in IDH-wildtype glioblastoma

Code for the survival analysis of IDH-wildtype glioblastoma that combines
clinical-molecular variables with shape features from three automatically
segmented tumour compartments (enhancing tumour, necrotic core, peritumoral
edema). It reproduces the tables and figures of the article: repeated nested
cross-validation of penalised Cox models, robustness and temporal analyses,
external validation of the frozen models, and the sensitivity analyses.

**Patient data are not included.** The scripts expect the input files
described below, which are available from the authors under the conditions
stated in the article.

## Requirements

Python 3.12. Install the dependencies with

```
python -m pip install -r requirements.txt
```

`pydicom` is needed only by `dicom_table_s1.py`; it is listed in
`requirements.txt` and runs in the same environment as the rest of the code.

## Configuration

Copy `private_config.example.json` to `private_config.json` and fill it in.
`private_config.json` is excluded from the repository. It holds:

- `paths`: location of the input files and of the output folders. Relative
  paths are resolved against the code folder. Every key is optional; the
  defaults are the names used in `config_cox.py`.
- `lost_to_followup_ids`: study IDs of the patients lost to follow-up, who are
  censored at their last contact.
- `exclude_ids`: study IDs excluded after review (empty in the article).
- `target_fixes`: corrections of the descriptive 12-month endpoint, as
  `{"study ID": value}`.
- `field_id_cutoff`: last study ID imaged on the 1.5 T system; later IDs were
  imaged at 3 T. It defines the batches for scanner harmonisation.

The configuration file can also be given through the environment variable
`GBM_PRIVATE_CONFIG`. All other settings (features, penalty grid, number of
repeats, permutations, horizons) are in `config_cox.py`.

## Files

| File | Content |
|---|---|
| `config_cox.py` | paths, cohort definitions, features, model and evaluation settings |
| `data_cox.py` | loading, recoding and scanner harmonisation (ComBat) |
| `pipelines_cox.py` | preprocessing transformers and the penalised Cox pipeline |
| `evaluation_cox.py` | repeated nested cross-validation, optimism correction, final model, model comparisons |
| `analyses_cox.py` | permutation test, risk stratification, calibration, SHAP, time-dependent AUC, orthogonality |
| `checks_cox.py` | proportional hazards, follow-up windows, crossover, time interaction, subcohorts, sample size, Monte Carlo, decision curve |
| `figures_cox.py` | all figures |
| `run_cox.py` | analysis stages of the derivation cohort |
| `run_pipeline.py` | runs every stage in order and writes a summary report |
| `freeze_model.py` | freezes the final models |
| `validate_external.py` | external validation of the frozen models |
| `model/frozen_model.json` | coefficients, preprocessing parameters and thresholds of the frozen models (aggregate values only) |
| `supplementary_checks.py` | potential follow-up, proportional hazards with the out-of-fold score, expected precision by cohort size (Figure S14) |
| `sensitivity_no_combat.py` | main results without scanner harmonisation |
| `sensitivity_external_raw.py` | external validation without harmonising the external features |
| `sensitivity_early_deaths.py` | analysis excluding early deaths |
| `sensitivity_p53.py` | alternative p53 coding |
| `volume_check.py` | radiomic score versus tumour volume |
| `scanner_check_shape.py` | field-strength differences of the analysed features |
| `complementarity_oof.py` | likelihood-ratio complementarity test from the checkpoints |
| `table1.py` | Table 1 |
| `dicom_table_s1.py` | Supplementary Table S1, MRI parameters from DICOM headers |
| `figure_flowchart.py` | Figure 1 |
| `redraw_main_figures.py` | Figures 2, 3A, 3B (preliminary) and 4 |
| `redraw_figure3b.py` | Figure 3B from the time-interaction model |
| `figure_external_forest.py` | Figure 4 |
| `redraw_supplementary_figures.py` | supplementary figures |

## Input data

Column names are those read by the code; no real values are shown.

**Derivation clinical file** (`.ods` or `.csv`): `ID_PAZIENTE`, `GENERE`
(`M`/`F`), `Età`, `Lobo` (free-text location), `RESEZIONE` (`GTR` or other),
`Morto`, `DATA_RMN_BASALE`, `DATA_DIAGNOSI_ISTOLOGICA`, `DATA_MORTE/ULTIMA`
(dates as `dd/mm/yyyy`), `MGMT_METILAZIONE` (%), `IDH_MUTAZIONE`, `Ki67_%`,
`p53_MUTAZIONE` (immunohistochemistry category: `assente`, `rare cellule`,
`debole`, `intermedia`, `intensa`, `diffusa`). An optional `target` column holds
the descriptive 12-month endpoint.

**Radiomic files** (derivation and external, `.csv`): one row per patient,
`ID_PAZIENTE` followed by the Quibim features named `<region>_<feature>`, with
region prefixes `enh`, `nec` and `ede`, for example `nec_Volume_of_Mesh`.

**External clinical file** (`.xlsx` or `.csv`): `ID_Studio` (numeric study ID,
optionally followed by a letter), `Date of surgery` (`dd/mm/yyyy`), `Età`,
`GENERE`, `RESEZIONE`, `MGMT_METILAZIONE` (0/1), `p53_ESPRESSIONE` (same
categories as above), `Ki67%`, `OS` (months), `Vivo` (`yes`/`no`).

**DICOM folders** for `dicom_table_s1.py`: one subfolder per patient, named
with the study ID. The optional `--cohort` file is a CSV with a column
`study_id` and, optionally, `field_T_used` (1.5 or 3.0), the field strength
used for harmonisation.

## Reproducing the results

Run from the code folder with the configured environment.

1. Main analysis, from data to external validation (several hours; every long
   stage resumes if interrupted):

   ```
   python run_pipeline.py
   ```

   The stages are: cohort description, nested cross-validation of the primary
   and secondary models, optimism correction and supplementary analyses,
   robustness checks, Monte Carlo cross-validation, permutation test, figures,
   model freezing, external validation and flowchart. A single stage can be
   rerun with `--only N`, or the run resumed with `--from-step N`. Stage 9
   rewrites `model/frozen_model.json`; to validate the published model, keep
   the file supplied here and skip that stage.

2. Additional analyses, after the pipeline:

   ```
   python supplementary_checks.py
   python table1.py --clinical <external clinical> --radiomic <external radiomic>
   python scanner_check_shape.py
   python complementarity_oof.py
   python sensitivity_no_combat.py
   python sensitivity_external_raw.py --clinical <external clinical> --radiomic <external radiomic>
   python sensitivity_early_deaths.py --months 2
   python sensitivity_p53.py
   python volume_check.py
   python dicom_table_s1.py --root <DICOM folder> --label Derivation --cohort <ids.csv> --group field
   ```

3. Final figures, in this order:

   ```
   python redraw_main_figures.py
   python redraw_figure3b.py
   python figure_external_forest.py
   python redraw_supplementary_figures.py
   ```

   `redraw_figure3b.py` must follow `redraw_main_figures.py`, which draws a
   preliminary version of Figure 3B.

Tables are written to `results_cox/` and figures to `figures_cox/`; external
validation results go to `results_cox/external_validation/`.

## Citation

If you use this code, please cite:

[Authors]. Shape radiomics and clinical-molecular variables provide complementary, time-dependent prognostic information in IDH-wildtype glioblastoma. Submitted to Frontiers in Oncology (2026).

Code: [Zenodo DOI]

## License

Released under the MIT License (see `LICENSE`).
