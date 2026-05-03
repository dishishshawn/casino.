"""Unit tests for casino.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from casino.config import CasinoConfig, get_config


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Isolate every test from any real .env on disk and from the singleton cache.
    get_config.cache_clear()
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith(("ANTHROPIC", "ALPACA", "TIINGO", "FMP", "DISCORD", "CASINO", "SEC_")):
            monkeypatch.delenv(k, raising=False)


def _write_env(tmp_path: Path, body: str) -> None:
    (tmp_path / ".env").write_text(body, encoding="utf-8")


def test_get_config_loads_from_env(tmp_path: Path) -> None:
    _write_env(
        tmp_path,
        "\n".join(
            [
                "ANTHROPIC_API_KEY=sk-anthropic",
                "ALPACA_API_KEY=ak",
                "ALPACA_SECRET_KEY=sk",
                "TIINGO_API_KEY=tk",
            ]
        ),
    )
    cfg = get_config()
    assert cfg.anthropic_api_key == "sk-anthropic"
    assert cfg.alpaca_api_key == "ak"
    assert cfg.alpaca_secret_key == "sk"
    assert cfg.tiingo_api_key == "tk"
    assert cfg.alpaca_base_url == "https://paper-api.alpaca.markets"


def test_optional_keys_default_none(tmp_path: Path) -> None:
    _write_env(
        tmp_path,
        "\n".join(
            [
                "ANTHROPIC_API_KEY=x",
                "ALPACA_API_KEY=x",
                "ALPACA_SECRET_KEY=x",
                "TIINGO_API_KEY=x",
            ]
        ),
    )
    cfg = get_config()
    assert cfg.fmp_api_key is None
    assert cfg.discord_webhook_url is None


def test_trading_params_defaults(tmp_path: Path) -> None:
    _write_env(tmp_path, "")
    cfg = CasinoConfig()
    assert cfg.max_risk_per_trade == pytest.approx(0.015)
    assert cfg.max_single_name == pytest.approx(0.10)
    assert cfg.max_gross_exposure == pytest.approx(1.0)
    assert cfg.kelly_fraction == pytest.approx(0.25)


def test_kelly_full_kelly_rejected() -> None:
    with pytest.raises(ValidationError):
        CasinoConfig(kelly_fraction=0.75)


def test_kelly_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        CasinoConfig(kelly_fraction=0.0)


def test_per_trade_risk_cap_enforced() -> None:
    with pytest.raises(ValidationError):
        CasinoConfig(max_risk_per_trade=0.05)


def test_get_config_is_cached(tmp_path: Path) -> None:
    _write_env(
        tmp_path, "ANTHROPIC_API_KEY=a\nALPACA_API_KEY=a\nALPACA_SECRET_KEY=a\nTIINGO_API_KEY=a\n"
    )
    a = get_config()
    b = get_config()
    assert a is b
