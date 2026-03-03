# Kimi API Baseline

API-based baseline using Kimi (kimi-k2.5) to predict building entrances from footprint geometry and map them to candidate entrance points.

## Files

- `client.py`: Kimi API client and cost estimation
- `prompt_builder.py`: Builds strict JSON-only prompts
- `postprocess.py`: Maps predicted entrances to candidate points
- `run_kimi_baseline.py`: End-to-end runner

## Setup

Environment variables:

- `MOONSHOT_API_KEY` (required)
- `MOONSHOT_API_BASE` (optional, default: https://api.moonshot.ai/v1)

<!-- `config.yaml` example:

```yaml
processed_data: data/processed
cities: ['nyc']
``` -->

## Run

From the repository root:

```bash
python -m src.baselines.kimi.run_kimi_baseline
```

## Outputs

- `outputs/kimi/preds_<timestamp>.csv`
- `outputs/kimi/eval_results_<timestamp>.json`

The script logs progress and the accumulated estimated API cost during execution.

Note: The script currently uses a mock dataset for testing. Replace it with `BuildingEntranceDataset` for real runs.
