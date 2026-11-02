# PV-BESS Downside Risk

Capacity planning and investment assessment of commercial and industrial photovoltaic–battery systems under persistent changes in electricity demand and contract prices.

The project considers an energy management contract in which an investor owns and operates the PV–battery system. Electricity supplied to the user is settled at a negotiated fraction of the grid time-of-use tariff. The investor pays for battery charging from the grid and receives export revenue and a share of demand-charge savings.

## Experiments

**Pre-contract planning.** PV-only and PV–battery investments are optimised across load–price conditions. Investment-entry and battery-upgrade boundaries separate the no-investment, PV-only and PV–battery regions. A supplementary analysis reoptimises capacities under 19 parameter settings.

**Fixed-capacity assessment.** The baseline capacities are held fixed across 1,000 Latin hypercube scenarios and 15 onset years. Each scenario represents a persistent load–price state from its onset year to the end of the project. The outputs include lifecycle investor NPV, changes from baseline NPV, user electricity costs and percentile summaries.

The case study uses hourly load and PV data from the Luoyuan expressway service area in Fujian, China, together with the local time-of-use tariff.

## Repository structure

```text
src/
  config.py        Model parameters and file paths
  data.py          Input processing and result storage
  model.py         Capacity planning, dispatch and financial accounting
  experiment1.py   Baseline planning, investment boundaries and sensitivity
  experiment2.py   LHS scenarios and fixed-capacity lifecycle assessment
  plots.py         Figures generated from saved results
  run.py           Command-line entry point
tests/             Unit and workflow tests
```

## Installation

Python 3.10 or later and a valid Gurobi licence are required for optimisation.

```bash
git clone https://github.com/Haoyu-0621/PV-BESS-Downside-Risk.git
cd PV-BESS-Downside-Risk
python -m pip install -r requirements.txt
```

## Input data

The four input workbooks are included in `inputs/`. File names, worksheet names and text labels are in English. Numeric observations and row order are unchanged from the original case-study inputs. Paths and the reference site name are set in `src/config.py`.

| File | Required layout |
|---|---|
| `luoyuan_load.xlsx` | `Load` worksheet: `Hour` contains indices 1–8760 and `Load (kW)` gives site demand. |
| `luoyuan_pv_output.xlsx` | `PV Output` worksheet: `Hour` contains indices 1–8760 and `PV output (kW)` gives reference PV generation. |
| `pv_reference_capacity.xlsx` | `PV Reference` worksheet: the first column identifies the site and the fifth gives reference PV capacity in kW. `Luoyuan Service Area` matches `PV_REF_NAME`. |
| `fuzhou_tou_tariff.xlsx` | `TOU Tariff` worksheet: 12 monthly rows, English month names in the first column, and 24 hourly tariffs in CNY/kWh in the following columns. |

Load, PV output and tariffs must refer to the same hourly sequence for a non-leap year. PV output is normalised by the reference installed capacity.

## Running the analysis

Run commands from the repository root:

```bash
# Baseline capacity and hourly dispatch
python -m src.run baseline

# Investment-entry and battery-upgrade boundaries
python -m src.run experiment1

# Fixed-capacity scenarios and onset-year assessment
python -m src.run experiment2

# One-at-a-time parameter sensitivity with capacity reoptimisation
python -m src.run sensitivity

# Figures from saved results; no optimisation is performed
python -m src.run plot
```

To generate the scenario set without solving the model:

```bash
python -m src.run experiment2 --prepare-only
```

Results are saved under `codex工作文件/实验结果/` by default. Calculation commands accept `--out`; the plotting command accepts `--root` for the result root and `--out` for figures. Run `python -m src.run --help` or add `--help` to a subcommand for details. Boundary calculations and scenario evaluations support resuming interrupted runs.

## Model settings

The baseline load factor is 1.0 and the contract-price factor is 0.8. The project horizon is 15 years. The main scenario set independently samples load factors from 0.25–1.0 and contract-price factors from 0.4–1.0, using seed `20260913`. Values above 0.8 provide higher-price comparisons. The equally weighted scenarios describe the selected test domain, rather than empirically estimated event probabilities.

Planning and fixed-capacity normal operation impose a 70% PV self-use requirement without an export-power limit. In the changed operating state, the self-use requirement is released and grid exports are capped at 1,000 kW. Degradation remains indexed to commissioning when cash flows are assembled across onset years.

Baseline planning and fixed-capacity assessments enforce charging–discharging exclusivity. The boundary command retains the LP relaxation used for the existing boundary analysis; its output metadata records this approximation. Figures are handled separately in `plots.py` and do not trigger model solves.

## Tests

```bash
python -m pytest -q tests -p no:cacheprovider
```

The tests use constructed inputs and mocked solver results. They can run without the original Excel files or a Gurobi licence; they do not reproduce the full optimisation experiments.

## Licence

[MIT](LICENSE).
