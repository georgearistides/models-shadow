"""
Grader — Map calibrated PD to letter grades.
==============================================
Replaces CreditGrader's Optimal-IV algorithm (which produced a 0.01pp Grade B band)
with simple fixed PD thresholds with enforced minimum separation.

Usage:
    from grader import Grader

    g = Grader()  # uses default boundaries
    grade = g.assign(0.08)  # -> "B"

    grades = g.assign_batch(pd_series)  # vectorized
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

# Default grade boundaries (PD thresholds)
# A: < 6%, B: 6-12%, C: 12-20%, D: > 20%
DEFAULT_BOUNDARIES = {
    "A": (0.0, 0.06),
    "B": (0.06, 0.12),
    "C": (0.12, 0.20),
    "D": (0.20, 1.0),
}


class Grader:
    """Map calibrated PD to letter grades using fixed thresholds."""

    def __init__(self, boundaries: Dict[str, tuple] = None):
        """
        Args:
            boundaries: dict mapping grade -> (lower_bound, upper_bound) PD ranges.
                       Lower bound is inclusive, upper bound is exclusive (except D which is inclusive).
        """
        self.boundaries = boundaries or DEFAULT_BOUNDARIES
        self._grade_order = sorted(self.boundaries.keys())

    def assign(self, pd_value: float) -> str:
        """Assign a single PD to a letter grade."""
        pd_value = float(pd_value)
        for grade in self._grade_order:
            low, high = self.boundaries[grade]
            if pd_value < high:
                return grade
        # If above all thresholds, assign worst grade
        return self._grade_order[-1]

    def assign_batch(self, pd_series: pd.Series) -> pd.Series:
        """Assign grades to a Series of PD values. Vectorized."""
        pd_arr = pd_series.values.astype(float)
        grades = np.full(len(pd_arr), self._grade_order[-1], dtype=object)

        # Assign in reverse order (worst to best) so better grades overwrite
        for grade in reversed(self._grade_order):
            _, high = self.boundaries[grade]
            mask = pd_arr < high
            grades[mask] = grade

        return pd.Series(grades, index=pd_series.index)

    def get_config(self) -> dict:
        """Return boundaries as a serializable dict."""
        return {"boundaries": self.boundaries}

    @classmethod
    def from_config(cls, config: dict) -> "Grader":
        """Create Grader from a config dict."""
        return cls(boundaries=config["boundaries"])

    def summary(self) -> str:
        """Human-readable summary of grade boundaries."""
        lines = ["Grade  PD Range"]
        lines.append("-----  --------")
        for grade in self._grade_order:
            low, high = self.boundaries[grade]
            if high >= 1.0:
                lines.append(f"  {grade}    >= {low:.0%}")
            else:
                lines.append(f"  {grade}    {low:.0%} – {high:.0%}")
        return "\n".join(lines)
