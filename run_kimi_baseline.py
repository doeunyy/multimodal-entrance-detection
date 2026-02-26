import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from shapely.geometry import Polygon

from src.baselines.kimi.client import call_kimi_api, estimate_cost_usd
from src.baselines.kimi.postprocess import map_to_candidates
from src.baselines.kimi.prompt_builder import build_entrance_prompt
from src.data_loader import BuildingEntranceDataset
from src.metrics import evaluate_predictions

load_dotenv()


# FOR TESTING PURPOSES ONLY: Mock dataset class
class MockDataset:
    def __init__(self, city, split, config_path):
        self.city = city
        # Dummy data matching the actual dataset schema
        self.data = [
            {
                "building_id": (city, 101),
                "geometry": Polygon([(0, 0), (0, 0.001), (0.001, 0.001), (0.001, 0)]),
                "candidates": [
                    (0.0005, 0.0),
                    (0.0, 0.0005),
                    (0.0005, 0.001),
                ],  # (lon, lat)
                "labels": [1, 0, 0],
            },
            {
                "building_id": (city, 102),
                "geometry": Polygon(
                    [(0.01, 0.01), (0.01, 0.011), (0.011, 0.011), (0.011, 0.01)]
                ),
                "candidates": [(0.0105, 0.01), (0.01, 0.0105)],
                "labels": [1, 0],
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def run_baseline():
    # Configuration Setup
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processed_path = config["processed_data"]
    cities = config.get("cities", ["nyc"])

    # Generate timestamp for unique file versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/kimi")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Versioned output filenames
    prediction_file = output_dir / f"preds_{timestamp}.csv"
    evaluation_file = output_dir / f"eval_results_{timestamp}.json"

    results_list = []
    total_cost = 0.0

    for city in cities:
        print(f"\n>>> Processing City: {city.upper()}")

        # NOTE: Uncomment the real Dataset loader and comment MockDataset for final run
        # dataset = BuildingEntranceDataset(city=city, split="test", config_path="config.yaml")

        # FOR TESTING PURPOSES ONLY
        dataset = MockDataset(city=city, split="test", config_path="config.yaml")

        for i in range(len(dataset)):
            sample = dataset[i]
            city_name, building_idx = sample["building_id"]

            # Step 1: Build Prompt using the 'geometry' Polygon (WKT)
            wkt_geometry = sample["geometry"].wkt
            prompt = build_entrance_prompt(wkt_geometry)

            # Step 2: API Call
            try:
                raw_json, usage = call_kimi_api(prompt, model="kimi-k2.5")
                total_cost += estimate_cost_usd(usage)
            except Exception as e:
                print(f"Error at {city_name}_{building_idx}: {e}")
                continue

            # Step 3: Map LLM results to Candidate Points
            candidate_probs = map_to_candidates(raw_json, sample["candidates"])

            # Step 4: Aggregate Results for Dataframe
            for c_idx, prob in enumerate(candidate_probs):
                results_list.append(
                    {
                        "city": city_name,
                        "building_idx": building_idx,
                        "candidate_idx": c_idx,
                        "predicted_prob": prob,
                    }
                )

            if (i + 1) % 10 == 0:
                print(
                    f"[{city}] {i+1}/{len(dataset)} buildings done. Acc. Cost: ${total_cost:.4f}"
                )

    # Save and Evaluate
    pred_df = pd.DataFrame(results_list)
    pred_df.to_csv(prediction_file, index=False)

    print(f"\nPredictions saved to: {prediction_file}")
    print(f"Total Estimated Cost: ${total_cost:.6f}")

    # Run metrics evaluation with versioned output path
    evaluate_predictions(
        processed_path=processed_path,
        predictions_file=str(prediction_file),
        cities=cities,
        output_file=str(evaluation_file),
    )
    print(f"Evaluation results saved to: {evaluation_file}")


if __name__ == "__main__":
    run_baseline()
