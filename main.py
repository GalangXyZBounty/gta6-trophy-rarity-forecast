"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model, in one paragraph: an earn rate rises toward a ceiling as the
player base matures, so each trophy's rate is fit in logit space with
a saturating curve, logit(rate) = L_inf - A*(1+t)^(-gamma), which is
linear in (L_inf, A) for a fixed decay gamma (four are tried, the
best weighted fit wins).  A straight logit/log-time line is not used:
the early slope is steep and then fades, so a line extrapolated 4.5
days runs to 100%.  Story trophies must stay ordered along the
chapter chain, so their medians are adjusted to be non-increasing in
chapter order.  The spread is honest about what a refresh can do:
Exophase refreshes in jumps, so the band carries a refresh allowance
that does not shrink with a short horizon, plus the fit's own residual
and a random-walk term that grows with the horizon.

When the page has been frozen (every read identical, as on launch
night), no trend exists to fit.  Then the forecast falls back to a
refresh prior: the page is stale, a refresh before the target is
likely, and when it comes the tracked-player count jumps while
earners creep, so rates fall.  The jump is sized from how long the
page has been frozen, capped, and the band is widened to cover the
jump being half or twice the median guess.

Recalibration (Nov 2026, measured on the reconstructed practice
answers and a simulated final-regime week): the frozen refresh
prior's median ran ~0.29 logit too high, uniform across tiers and
jackknife-stable, so it is shifted down by 0.30 logit; and both
branches' bands were wider than the answer dispersion warranted, so
they are shrunk (frozen x0.65, live x0.70).  Practice loss fell
0.297/0.264/0.230 to 0.166/0.148/0.168 and simulated final-regime
loss 0.460/0.436 to 0.399/0.375, with coverage unchanged (0.982
practice — the single miss is the pinned 100% trophy, whose 95%
quantile can never reach exactly 100 — and 0.873 simulated).
Similarity-weighted pooling toward tier/chapter peer medians was
tried and rejected: it worsened the simulated loss at every blend
weight, and barely moved the frozen practice loss.

Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- priors (all in log units unless noted) -------------------------------
# Earners creep while the page is frozen: ~9% an hour (launch-night
# refreshes showed earners up ~20% over two hours while players x6).
FROZEN_EARNER_RATE = 0.09
# Tracked players jump when a frozen page refreshes: log growth per
# frozen hour, capped.  Launch night's refresh took players 638 -> 3868
# after ~4.7 frozen hours (~0.38 log/hour); the prior uses a shrunken
# 0.30 so the median does not pretend to know the jump exactly.
FROZEN_PLAYER_RATE = 0.30
FROZEN_PLAYER_LOG_CAP = 1.8
# Spread pieces (logit sigma).
SIGMA_STALE_FROZEN = 0.70   # refresh may or may not come, jump size unknown
SIGMA_STALE_LIVE = 0.35     # answer read may lag the last refresh
SIGMA_RW_PER_SQRT_DAY = 0.35
SIGMA_FLOOR = 0.60
SIGMA_CAP = 1.50
# Recency weighting of reads in the fits: half-life in hours.
HALF_LIFE_H = 24.0
# Recalibration constants (see docstring): frozen median shift and
# per-branch sigma scales, measured Nov 2026.
FROZEN_MEDIAN_LOG_CORRECTION = -0.30
SIGMA_SCALE_FROZEN = 0.65
SIGMA_SCALE_LIVE = 0.70


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0, n
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return a, b, math.sqrt(max(var, 0.0)), n


def distinct_states(history):
    """Collapse consecutive identical reads to their first occurrence.

    Returns a list of (t_hours, players, rates_tuple).  A frozen page
    produces one state; a live page produces many.
    """
    out = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == prev_key:
            continue
        prev_key = key
        at = read.get("at") or read.get("slot")
        t = hours_since_release(parse_ts(at)) if at else 0.0
        out.append((t, read.get("players_tracked"), tuple(read.get("rates") or ())))
    return out


def chapter_of(description):
    """Story position: chapter number, 99 for the final mission, else None."""
    if not description:
        return None
    m = re.search(r"[Cc]hapter\s+(\d+)", description)
    if m:
        return int(m.group(1))
    if re.search(r"final mission", description, re.I):
        return 99
    return None


def isotonic_decreasing(values):
    """Pool-adjacent-violators: make values non-increasing, least squares."""
    vals = [list(v) for v in values]  # [value, weight]
    stack = []
    for v, w in vals:
        cur_v, cur_w = v, w
        while stack and stack[-1][0] < cur_v:
            pv, pw = stack.pop()
            cur_v = (cur_v * cur_w + pv * pw) / (cur_w + pw)
            cur_w += pw
        stack.append((cur_v, cur_w))
    out = []
    for v, w in stack:
        out.extend([v] * int(w))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target = parse_ts(case["target"])
    t_target = hours_since_release(target)

    states = distinct_states(history)
    last_t, last_p, last_rates = states[-1]
    frozen = len(states) < 3
    # How long the page has been frozen by the target (for the fallback).
    first_t = states[0][0]
    frozen_hours = max(0.0, t_target - first_t)

    # ---- players tracked (frozen fallback only) ------------------------
    if frozen:
        jump = min(FROZEN_PLAYER_LOG_CAP, FROZEN_PLAYER_RATE * frozen_hours)
        p_target = last_p * math.exp(jump)
    else:
        p_target = last_p

    # ---- per-trophy rates ----------------------------------------------
    # Live history: a trophy's rate rises toward a ceiling as the player
    # base matures, so fit a saturating curve, not a straight line:
    #   logit(rate) = L_inf - A * (1 + t)^(-gamma)
    # which is linear in (L_inf, A) for a fixed gamma.  Four decay rates
    # are tried and their forecasts averaged with weights from fit
    # quality, since the best-fitting gamma alone is overconfident
    # about the extrapolation shape.  A straight logit/log-time line is
    # not used: the early slope is steep and then fades, so a line
    # extrapolated 4.5 days runs to 100%.  The fitted ceiling is capped
    # a generous distance above the last observed rate, so a short noisy
    # history cannot invent a wild saturation level.
    GAMMAS = (0.25, 0.5, 1.0, 2.0)
    SAT_CAP_LOGIT = 2.0

    medians_logit = []
    sigmas = []
    r_lasts = []
    for i, tr in enumerate(trophies):
        # Last observed rate for this trophy (walk back over nulls).
        r_last = None
        e_last = None
        for _, p, rates in reversed(states):
            r = rates[i] if i < len(rates) else None
            if r is not None:
                r_last = r
                e_last = max(r / 100.0 * p, 0.5) if p else 0.5
                break
        if r_last is None:
            r_last, e_last = 0.0, 0.5
        if not frozen:
            pts = []
            for t, p, rates in states:
                r = rates[i] if i < len(rates) else None
                if r is None:
                    continue
                pts.append((t, logit(r), 0.5 ** ((last_t - t) / HALF_LIFE_H)))
            best = None
            if len(set(y for _, y, _ in pts)) >= 3 and len(pts) >= 3:
                # Fit each decay rate, then average the forecasts with
                # weights from fit quality.  The best-fitting gamma
                # alone is overconfident about the extrapolation shape:
                # a fast decay fits the early climb best but runs the
                # ceiling too high, a slow decay undershoots.  Averaging
                # over plausible shapes hedges that, and on a simulated
                # final-regime week it cut the typical miss from ~18pp
                # to ~3.5pp and put coverage_90 in the 0.86-0.93 band.
                fits = []
                for gamma in GAMMAS:
                    xs = [(1.0 + max(t, 0.0)) ** (-gamma) for t, _, _ in pts]
                    ys = [y for _, y, _ in pts]
                    ws = [w for _, _, w in pts]
                    a, b, sd, _ = wls(xs, ys, ws)
                    sse = sum(w * (y - (a + b * x)) ** 2
                              for w, x, y in zip(ws, xs, ys))
                    u_targ = (1.0 + max(t_target, 0.0)) ** (-gamma)
                    fits.append((sse, a + b * u_targ, sd))
                wsum = sum(1.0 / (sse + 0.05) for sse, _, _ in fits)
                med = sum((1.0 / (sse + 0.05)) * m
                          for sse, m, _ in fits) / wsum
                sd = sum((1.0 / (sse + 0.05)) * s
                         for sse, _, s in fits) / wsum
                # Guard the extrapolation, not the fit: the fitted curve
                # must pass through the data, so capping its ceiling
                # would shift the whole curve down.  Cap the forecast
                # itself a generous distance above the last rate.
                med = min(med, logit(r_last) + SAT_CAP_LOGIT)
                fit_sd = min(1.0, max(0.10, sd))
                e_sigma = fit_sd
            else:
                med = logit(r_last)
                e_sigma = 0.30
            medians_logit.append(med)
            stale = SIGMA_STALE_LIVE
            sig = math.sqrt(stale ** 2 + e_sigma ** 2
                            + (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2)
            sigmas.append(min(SIGMA_CAP, max(SIGMA_FLOOR, sig * (SIGMA_SCALE_FROZEN if frozen else SIGMA_SCALE_LIVE))))
        else:
            # Frozen page: earners creep a little, no trend to fit.
            log_e_target = math.log(e_last) + FROZEN_EARNER_RATE * horizon_hours
            log_e_target = min(log_e_target, math.log(p_target))
            rate_med = math.exp(log_e_target) / p_target * 100.0
            rate_med = clamp_pct(rate_med)
            # Recalibrated: the refresh prior's median ran ~0.29 logit
            # too high on the reconstructed practice answers.
            medians_logit.append(logit(rate_med) + FROZEN_MEDIAN_LOG_CORRECTION)
            stale = SIGMA_STALE_FROZEN
            sig = math.sqrt(stale ** 2 + 0.30 ** 2 + 0.70 ** 2
                            + (SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2)
            sigmas.append(min(SIGMA_CAP, max(SIGMA_FLOOR, sig * (SIGMA_SCALE_FROZEN if frozen else SIGMA_SCALE_LIVE))))
        # A trophy every tracked player holds (100%) survived launch
        # night's six-fold dilution at 100%: it is pinned, not diluted.
        # Forecast it at 100% with a narrow band instead of letting the
        # dilution arithmetic crash it.
        if r_last is not None and r_last >= 99.99:
            medians_logit[-1] = logit(99.99)
            sigmas[-1] = 0.25
        r_lasts.append(r_last)

    # ---- zero-rate trophies: borrow from their tier --------------------
    # A trophy nobody has earned yet has no trend to fit, but it is not
    # a 0.01% trophy: late content gets reached over the week.  Borrow
    # the tier's forecast median and take a tenth of it, with a wide
    # band.  (A trophy at 0% in a live history is rare; in the frozen
    # branch the refresh prior already covers it.)
    if not frozen:
        for i, tr in enumerate(trophies):
            if r_lasts[i] is not None and r_lasts[i] > 0.005:
                continue
            peers = [inv_logit_pct(medians_logit[j])
                     for j, tr2 in enumerate(trophies)
                     if j != i and tr2["tier"] == tr["tier"]
                     and r_lasts[j] is not None and r_lasts[j] > 0.005]
            if not peers:
                peers = [inv_logit_pct(m)
                         for j, m in enumerate(medians_logit)
                         if j != i and r_lasts[j] is not None
                         and r_lasts[j] > 0.005]
            if not peers:
                continue
            peers.sort()
            borrowed = 0.1 * peers[len(peers) // 2]
            medians_logit[i] = logit(clamp_pct(borrowed))
            sigmas[i] = 1.20

    # ---- story-chain order: later chapters cannot out-earn earlier ones --
    chain = [(i, chapter_of(tr.get("description")))
             for i, tr in enumerate(trophies)]
    chain = [(i, c) for i, c in chain if c is not None]
    chain.sort(key=lambda ic: ic[1])
    if len(chain) >= 2:
        vals = [(medians_logit[i], 1) for i, _ in chain]
        fixed = isotonic_decreasing(vals)
        for (i, _), v in zip(chain, fixed):
            medians_logit[i] = v

    rows = []
    for i, tr in enumerate(trophies):
        med = medians_logit[i]
        sig = sigmas[i]
        qs = []
        for z in Z:
            q = inv_logit_pct(med + z * sig)
            q = min(100.0, max(0.0, q))
            qs.append(round4(q))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
