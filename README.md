# GTA VI Trophy Rarity Forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezhanime1/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezhanime1/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy — built for the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

The model fits a power law to how earners and players-tracked grow from release, shrinks each trophy's growth slope toward a prior learned from launch night, and widens its spread with fit residual and refresh-jump uncertainty. Local backtest: pinball loss 0.2133 vs the house baseline's 2.3800.

Run it with `python3 main.py` (reads a case from `$MAKEFUN_INPUT_DIR`, writes the forecast to `$MAKEFUN_OUTPUT_DIR`).
