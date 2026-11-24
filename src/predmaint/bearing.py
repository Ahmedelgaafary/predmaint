"""Rolling-element bearing kinematics: characteristic fault frequencies."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Defaults match the SKF 6205-2RS drive-end bearing used in the public CWRU dataset."""

    n_balls: int = 9
    ball_d: float = 7.94  # mm
    pitch_d: float = 39.04  # mm
    contact_angle_deg: float = 0.0

    @property
    def _r(self) -> float:
        return self.ball_d / self.pitch_d * math.cos(math.radians(self.contact_angle_deg))

    def fault_orders(self) -> dict[str, float]:
        """Fault frequencies as multiples of shaft frequency (orders)."""
        r = self._r
        return {
            "bpfo": self.n_balls / 2 * (1 - r),  # ball pass, outer race
            "bpfi": self.n_balls / 2 * (1 + r),  # ball pass, inner race
            "bsf": self.pitch_d / (2 * self.ball_d) * (1 - r**2),  # ball spin
            "ftf": 0.5 * (1 - r),  # cage (fundamental train)
        }


DEFAULT_GEOMETRY = BearingGeometry()


def fault_frequencies(rpm: float, geom: BearingGeometry = DEFAULT_GEOMETRY) -> dict[str, float]:
    """Fault frequencies in Hz at a given shaft speed."""
    fr = rpm / 60.0
    return {"fr": fr, **{k: v * fr for k, v in geom.fault_orders().items()}}
