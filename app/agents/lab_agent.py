"""Laboratory Analysis Agent.

Compares supplied laboratory values with the project's approximate,
educational reference ranges. Out-of-range values are *observations*,
not diagnoses.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.schemas import PatientCase
from app.utils.validators import is_valid_number, normalize_term

DEFAULT_RANGES_PATH = Path(__file__).resolve().parents[2] / "data" / "lab_reference_ranges.json"

LAB_NOTE = (
    "Reference-range comparison only. An out-of-range value is an observation "
    "that clinicians interpret in context; it is not a diagnosis."
)


def _fmt(value: float) -> str:
    """13500.0 -> '13500', 9.8 -> '9.8'."""
    return str(int(value)) if float(value).is_integer() else str(value)


class LaboratoryAnalysisAgent:
    """Classifies each lab value as low / normal / high / invalid / unknown."""

    def __init__(self, ranges_path: Optional[Path] = None):
        path = Path(ranges_path) if ranges_path else DEFAULT_RANGES_PATH
        with open(path, encoding="utf-8") as fh:
            self.tests: Dict[str, dict] = json.load(fh)["tests"]
        self._alias_index = {
            normalize_term(alias): key
            for key, spec in self.tests.items()
            for alias in spec.get("aliases", []) + [key]
        }

    def resolve_test(self, name: str) -> Optional[str]:
        """Map a user-supplied test name to a canonical reference key."""
        return self._alias_index.get(normalize_term(name))

    def _range_for(self, key: str, sex: str) -> Tuple[float, float]:
        spec = self.tests[key]
        by_sex = spec.get("by_sex", {})
        sex_key = normalize_term(sex)
        if sex_key in by_sex:
            low, high = by_sex[sex_key]
            return low, high
        return spec["low"], spec["high"]

    def evaluate_value(self, name: str, value, sex: str = "") -> dict:
        key = self.resolve_test(name)
        result = {"test": key or name, "input_name": name, "value": value}

        if not is_valid_number(value) or value < 0:
            result.update(status="invalid", note="Value is missing, non-numeric or negative; excluded from analysis.")
            return result
        if key is None:
            result.update(status="unknown", note="No reference range available in this educational dataset.")
            return result

        low, high = self._range_for(key, sex)
        spec = self.tests[key]
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        else:
            status = "normal"
        result.update(
            name=spec["name"],
            unit=spec["unit"],
            reference_range=f"{low}-{high} {spec['unit']}",
            status=status,
            note=f"{status.capitalize()} relative to the educational reference range.",
        )
        return result

    def analyze(self, patient: PatientCase) -> dict:
        results: List[dict] = [
            self.evaluate_value(name, value, patient.sex)
            for name, value in patient.lab_results.items()
        ]
        abnormal = [r for r in results if r["status"] in ("low", "high")]
        invalid = [r for r in results if r["status"] == "invalid"]
        unknown = [r for r in results if r["status"] == "unknown"]

        if not results:
            summary = "No laboratory results were supplied."
        else:
            summary = (
                f"{len(results)} lab value(s) reviewed: {len(abnormal)} outside the "
                f"reference range, {len(invalid)} invalid, {len(unknown)} without a reference range."
            )

        return {
            "patient_id": patient.patient_id,
            "results": results,
            "abnormal_findings": [f"{r['test']} {r['status']} ({_fmt(r['value'])} {r['unit']})" for r in abnormal],
            "invalid_values": [r["input_name"] for r in invalid],
            "summary": summary,
            "note": LAB_NOTE,
        }
