"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target as five quantiles, scored by pinball loss on the
logit scale.

Model (per trophy, all from the case's own history at run time):
  rate = earners / players_tracked.  Both grow roughly as powers of
  (hours since release + 1.2h):  N(t) ~ (t+1.2)^alpha,  E_t(t) ~ (t+1.2)^beta.
  The rate's exponent gamma = beta - alpha is negative (dilution beats
  progression early) and larger for trophies new players earn quickly.
  With >= 2 distinct reads we fit alpha and each beta from the history,
  anchoring every fit at the last read; with a flat history we fall back to
  a prior for gamma fitted to the launch-night -> day-2 trajectory of this
  game's page (a trophy's launch rate predicts its drop: fast trophies
  hold, late ones collapse).  The spread combines the fit's prediction
  uncertainty with the refresh-jump uncertainty seen in the history
  (Exophase refreshes its counts in jumps, hours apart), floored at the
  house baseline's random-walk spread.

Standard library only; deterministic; no network. Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release: 2026-11-19T00:00:00Z.  Times are hours since then; the +1.2h
# shift keeps log-time finite at the first reads (just before midnight).
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
SHIFT = 1.2

# Prior for the players-tracked growth exponent (log-log slope of N vs
# time): 638 players at launch, 3,868 at +4.5h on launch night.
ALPHA_PRIOR = 1.15
# Prior for a trophy's rate exponent gamma = beta - alpha, as a function of
# its rate at the first read: gamma ~ G0 + G1 * log10(rate).  Fitted to the
# launch-night -> 38h trajectory of this game's page (fast trophies hold
# their rate, late ones fall ~20x).  Clipped to [-0.95, 0].  A
# description-based shift accounts for time-to-earn: a Chapter 1 or
# quick-activity trophy is earned in a player's first hour, so it holds
# better than its launch rate alone suggests; a late chapter falls faster.
G0 = -0.898
G1 = 1.364
GAMMA_MIN = -1.35
GAMMA_MAX = 0.0
# Prior refresh-jump size (logit SD of one Exophase refresh, detrended) when
# the history shows too few refreshes to measure it.
JUMP_PRIOR = 0.6
GAP_PRIOR_H = 6.0
# Prior residual SD of log-earners around the power-law fit.
FIT_SD_PRIOR = 0.35
# Overall spread scale (tuned on the launch-night -> day-2 backtest; 0.3 is
# the backtest optimum, 0.4 keeps a little extra margin for the final's
# longer horizon).
SPREAD_SCALE = 0.4


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(s):
    return (parse_time(s) - T0).total_seconds() / 3600.0


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def linfit(xs, ys):
    """Least-squares y = a + b*x; returns (a, b).  xs, ys non-empty."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def gamma_prior(rate_first, description=""):
    g = G0 + G1 * math.log10(max(rate_first, 0.05))
    d = description.lower()
    if "chapter 1" in d:
        g += 0.55
    if "arcade" in d or "fishing" in d:
        g += 0.50
    if "chapter 3" in d or "chapter 4" in d or "final mission" in d:
        g -= 0.12
    return min(GAMMA_MAX, max(GAMMA_MIN, g))


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0
    t_cp = hours_since_release(case["at"])
    t_tg = hours_since_release(case["target"])
    x_tg = math.log(max(t_tg, -1.0) + SHIFT)
    n_tr = len(trophies)

    # --- distinct reads (Exophase refreshes are the only changes) ---
    seen = {}
    distinct = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key not in seen:
            seen[key] = True
            distinct.append(read)
    xs = [math.log(max(hours_since_release(r["at"]), -1.0) + SHIFT) for r in distinct]
    ns = [r["players_tracked"] for r in distinct]
    log_n = [math.log(max(n, 1)) for n in ns]

    # --- players-tracked growth: log N = a + alpha * x ---
    # Both N and the earners are extrapolated from the LAST DISTINCT read
    # (the most recent refresh): the level comes from that read, the fit
    # only sets the slope.  (Anchoring N at the first read while anchoring
    # E at the last read double-counts the N growth in between, which the
    # flat launch-night history directly contradicts: N stayed 638.)
    x_anchor = xs[-1]
    n_anchor = ns[-1]
    if len(distinct) >= 2:
        _, alpha = linfit(xs, log_n)
        alpha = min(2.0, max(0.3, alpha))
    else:
        alpha = ALPHA_PRIOR
    n_tg = n_anchor * math.exp(alpha * (x_tg - x_anchor))
    n_tg = max(n_tg, n_anchor)
    x_last_read = math.log(max(hours_since_release(history[-1]["at"]), -1.0) + SHIFT)
    span = max(0.5, x_last_read - xs[0])

    # --- refresh jumps in logit(rate), pooled over trophies, detrended by
    # the fitted rate trend (so the trend itself does not inflate the noise) ---
    jumps = []
    for j in range(1, len(distinct)):
        prev, cur = distinct[j - 1]["rates"], distinct[j]["rates"]
        dt_x = xs[j] - xs[j - 1]
        for i in range(n_tr):
            if prev[i] is not None and cur[i] is not None and prev[i] > 0 and cur[i] > 0:
                trend = gamma_prior(prev[i]) * dt_x
                jumps.append((logit(cur[i]) - logit(prev[i])) - trend)
    if len(distinct) < 2:
        # Flat history: the next refresh is one discrete jump, and its size
        # is the whole story (launch night's refresh moved logit by ~1-3).
        jump_sd = 1.2
    elif len(jumps) >= 3:
        mj = sum(jumps) / len(jumps)
        jump_sd = math.sqrt(sum((j - mj) ** 2 for j in jumps) / (len(jumps) - 1))
        jump_sd = min(1.5, max(0.15, jump_sd))
    else:
        jump_sd = 0.6
    taus = [max(hours_since_release(r["at"]), -1.0) for r in distinct]
    gaps_h = [taus[j] - taus[j - 1] for j in range(1, len(taus)) if taus[j] > taus[j - 1]]
    if gaps_h:
        gap_h = max(1.0, min(24.0, sorted(gaps_h)[len(gaps_h) // 2]))
    else:
        gap_h = GAP_PRIOR_H
    n_ref = max(1.0, horizon_h / gap_h)
    s_refresh = jump_sd * math.sqrt(n_ref)

    # --- per-trophy fit residual SD (pooled) ---
    resid = []
    for i in range(n_tr):
        pts = []
        for j, r in enumerate(distinct):
            rate = r["rates"][i]
            if rate is not None and rate > 0:
                pts.append((xs[j], math.log(rate * r["players_tracked"] / 100.0)))
        if len(pts) >= 3:
            _, b = linfit([p[0] for p in pts], [p[1] for p in pts])
            a = sum(p[1] for p in pts) / len(pts) - b * sum(p[0] for p in pts) / len(pts)
            resid.extend(p[1] - (a + b * p[0]) for p in pts)
    if len(resid) >= 5:
        mr = sum(resid) / len(resid)
        fit_sd = math.sqrt(sum((v - mr) ** 2 for v in resid) / (len(resid) - 1))
        fit_sd = min(1.0, max(0.1, fit_sd))
    else:
        fit_sd = FIT_SD_PRIOR

    floor_s = 0.5 * math.sqrt(max(horizon_h, 0.0) / 24.0)

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = distinct[-1]["rates"][i]
        rate_first = next((r["rates"][i] for r in distinct if r["rates"][i] is not None), None)
        if rate_last is None:
            rate_last = rate_first if rate_first is not None else 0.0
        if rate_first is None:
            rate_first = rate_last

        if rate_last >= 99.0:
            # The everyone-has-it trophy: stays ~100%, tiny spread.
            med = min(99.99, max(99.9, rate_last - 0.01))
            s = 0.4
        else:
            # Earners at each distinct read; fit log E on x, anchored at
            # the last read (the level comes from the last read, the fit
            # only sets the slope, shrunk toward the gamma prior).
            pts = []
            for j, r in enumerate(distinct):
                rate = r["rates"][i]
                if rate is not None and rate > 0:
                    pts.append((xs[j], math.log(rate * r["players_tracked"] / 100.0)))
            gp = gamma_prior(rate_first, trophy.get("description", ""))
            if len(pts) >= 2:
                _, beta_fit = linfit([p[0] for p in pts], [p[1] for p in pts])
                gamma_fit = beta_fit - alpha
                w = len(pts) / (len(pts) + 2.0)
                gamma = w * gamma_fit + (1 - w) * gp
            else:
                gamma = gp
            beta = gamma + alpha
            e_last = max(rate_last, 0.01) * ns[-1] / 100.0
            log_e_tg = math.log(max(e_last, 1e-6)) + beta * (x_tg - x_anchor)
            e_tg = math.exp(log_e_tg)
            med = clamp(100.0 * e_tg / n_tg)
            # Prediction uncertainty grows with extrapolation distance.
            extra = max(0.0, x_tg - x_anchor) / span
            s_pred = fit_sd * math.sqrt(1.0 + extra * extra)
            s = math.sqrt(s_pred * s_pred + s_refresh * s_refresh)
            s = min(2.2, max(floor_s, s))
            s *= SPREAD_SCALE
            s = min(2.2, max(0.15, s))

        lg = logit(med)
        qs = [round4(clamp(inv_logit(lg + z * s))) for z in Z]
        # enforce non-decreasing (rounding can tie, never invert)
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
