"""Cost accounting in one currency (USD) with the assumptions spelled out.

API models: Copilot AI units (AIU) from the SDK usage event, converted at ``usd_per_aiu`` (default 0.01, i.e. one
AIU = one US cent; override with ``BENCH_USD_PER_AIU``). Local models, two views:
  energy  measured GPU energy (kWh) x electricity tariff (``BENCH_TARIFF_PER_KWH`` + ``BENCH_TARIFF_CURRENCY``);
          left as None ("not computed") until a tariff is given
  rental  wall-clock GPU time x on-demand price of the same GPU class (``BENCH_GPU_USD_PER_HOUR``), i.e. what the
          same run would cost on a rented L40S
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_float(name: str) -> Optional[float]:
    v = os.getenv(name, "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


@dataclass
class Prices:
    usd_per_aiu: float = 0.01
    tariff_per_kwh: Optional[float] = None
    tariff_currency: str = ""
    gpu_usd_per_hour: Optional[float] = None
    gpu_price_source: str = ""

    @classmethod
    def from_env(cls) -> "Prices":
        return cls(usd_per_aiu=_env_float("BENCH_USD_PER_AIU") or 0.01,
                   tariff_per_kwh=_env_float("BENCH_TARIFF_PER_KWH"),
                   tariff_currency=os.getenv("BENCH_TARIFF_CURRENCY", "").strip(),
                   gpu_usd_per_hour=_env_float("BENCH_GPU_USD_PER_HOUR"),
                   gpu_price_source=os.getenv("BENCH_GPU_PRICE_SOURCE", "").strip())

    def api_usd(self, aiu: Optional[float]) -> Optional[float]:
        return None if aiu is None else aiu * self.usd_per_aiu

    def energy_cost(self, kwh: Optional[float]) -> Optional[float]:
        """Electricity cost in ``tariff_currency``; None while the tariff is unknown."""
        if kwh is None or self.tariff_per_kwh is None:
            return None
        return kwh * self.tariff_per_kwh

    def rental_usd(self, gpu_seconds: Optional[float]) -> Optional[float]:
        if gpu_seconds is None or self.gpu_usd_per_hour is None:
            return None
        return gpu_seconds / 3600.0 * self.gpu_usd_per_hour

    def describe(self) -> dict:
        return {"usd_per_aiu": self.usd_per_aiu, "tariff_per_kwh": self.tariff_per_kwh,
                "tariff_currency": self.tariff_currency or None, "gpu_usd_per_hour": self.gpu_usd_per_hour,
                "gpu_price_source": self.gpu_price_source or None}


def joules_to_kwh(j: Optional[float]) -> Optional[float]:
    return None if j is None else j / 3.6e6
