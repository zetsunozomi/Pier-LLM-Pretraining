from __future__ import annotations

import json
from pathlib import Path


PARENT_DIR = Path(
	"/pscratch/sd/s/syfan/Pier/downstream_task_workpath/results/checkpoints_exp1_50_new_new"
)

TASK_ORDER = [
	("boolq", "acc,none"),
	("cb", "acc,none"),
	("copa", "acc,none"),
	("multirc", "acc,none"),
	("record", "f1,none"),
	("sglue_rte", "acc,none"),
	("wic", "acc,none"),
	("wsc", "acc,none"),
	("lambada_openai", "acc,none"),
	("race", "acc,none"),
	("mathqa", "acc,none"),
	("piqa", "acc,none"),
	("winogrande", "acc,none"),
]


def find_model_dir(parent_dir: Path) -> str:
	for task_name, _ in TASK_ORDER:
		task_dir = parent_dir / task_name
		if not task_dir.is_dir():
			continue
		for child in sorted(task_dir.iterdir()):
			if child.is_dir():
				return child.name
	raise FileNotFoundError(f"No model directory found under {parent_dir}")


def load_task_score(parent_dir: Path, model_dir: str, task_name: str, metric_name: str) -> float:
	result_dir = parent_dir / task_name / model_dir
	json_files = sorted(result_dir.glob("results_*.json"))
	if not json_files:
		raise FileNotFoundError(f"No result JSON found in {result_dir}")

	data = json.loads(json_files[0].read_text())
	metric_value = data["results"][task_name][metric_name]
	return float(metric_value)


def extract_scores(parent_dir: Path) -> list[tuple[Path, str, float]]:
	scores: list[tuple[Path, str, float]] = []

	for json_path in sorted(parent_dir.rglob("*.json")):
		try:
			data = json.loads(json_path.read_text())
		except json.JSONDecodeError:
			continue

		results = data.get("results", {})
		for task_name, task_results in results.items():
			for metric_name, metric_value in task_results.items():
				if not isinstance(metric_value, (int, float)):
					continue
				if not metric_name.startswith("acc"):
					continue

				scores.append((json_path.relative_to(parent_dir), task_name, float(metric_value)))

	return scores


def main() -> None:
	model_dir = find_model_dir(PARENT_DIR)
	scores = [load_task_score(PARENT_DIR, model_dir, task_name, metric_name) for task_name, metric_name in TASK_ORDER]
	formatted_scores = " & ".join(f"{score:.4f}" for score in scores)
	print("NAME & {} \\\\".format(formatted_scores))


if __name__ == "__main__":
	main()
