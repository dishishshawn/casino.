"""Centralized configuration.

All environment variables load here, validated, with typed accessors. Feature
modules MUST NOT call `os.environ` directly — import `get_config()` instead.

Hard rules from PRD §8 are enforced as field validators where they reduce to
single-value invariants (e.g., kelly_fraction ≤ 0.5).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CasinoConfig(BaseSettings):
    """Typed configuration loaded from `.env` and environment variables.

    PRD §10 conventions: money is Decimal, times are UTC, no os.environ in
    feature modules. The trading parameters defaulted here are guard-rails;
    the runtime sizing helpers in `execution/risk.py` enforce them on every
    order.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Required API keys ----
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    tiingo_api_key: str = Field(default="", alias="TIINGO_API_KEY")

    # ---- Defaulted endpoints ----
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL",
    )

    # ---- Optional API keys ----
    fmp_api_key: str | None = Field(default=None, alias="FMP_API_KEY")
    discord_webhook_url: str | None = Field(default=None, alias="DISCORD_WEBHOOK_URL")

    # ---- Storage paths ----
    data_dir: Path = Field(default=Path("data"), alias="CASINO_DATA_DIR")
    duckdb_path: Path = Field(default=Path("data/casino.duckdb"), alias="CASINO_DUCKDB_PATH")
    state_sqlite_path: Path = Field(
        default=Path("data/state.sqlite"),
        alias="CASINO_STATE_SQLITE_PATH",
    )

    # ---- SEC EDGAR fair-access ----
    sec_user_agent: str = Field(
        default="casino-trading contact@example.com",
        alias="SEC_USER_AGENT",
    )

    # ---- Trading parameters (PRD §8 hard rules) ----
    max_risk_per_trade: float = Field(default=0.015, ge=0.0, le=0.05)  # 1.5% NAV
    max_single_name: float = Field(default=0.10, ge=0.0, le=0.25)  # 10% NAV
    max_gross_exposure: float = Field(default=1.0, ge=0.0, le=1.5)  # 100% NAV
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=0.5)  # fractional Kelly

    # ---- Cost-model knobs ----
    transaction_cost_bps: float = Field(default=7.5, ge=0.0)  # round-trip on liquid US equities

    # ---- LLM models ----
    llm_default_model: str = Field(default="claude-sonnet-4-5")
    llm_haiku_model: str = Field(default="claude-haiku-4-5")
    llm_opus_model: str = Field(default="claude-opus-4-5")

    @field_validator("kelly_fraction")
    @classmethod
    def _kelly_must_be_fractional(cls, v: float) -> float:
        # PRD §8 rule 6: Fractional Kelly (¼ to ½) only. Never full Kelly.
        if v <= 0.0 or v > 0.5:
            raise ValueError("kelly_fraction must be in (0, 0.5]; full Kelly is forbidden")
        return v

    @field_validator("max_risk_per_trade")
    @classmethod
    def _per_trade_cap(cls, v: float) -> float:
        # PRD §8 rule 1: never risk more than 1.5% per trade.
        if v > 0.015 + 1e-9:
            raise ValueError("max_risk_per_trade must be ≤ 0.015 (1.5% of NAV)")
        return v


@lru_cache(maxsize=1)
def get_config() -> CasinoConfig:
    """Return the process-wide singleton configuration.

    Cleared in tests via `get_config.cache_clear()`.
    """
    return CasinoConfig()
