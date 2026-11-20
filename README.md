[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GalangXyZBounty/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GalangXyZBounty/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every one of the 55 GTA VI PlayStation 5 trophies, its Exophase earn rate on 2026-11-26 as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

## How it works

Each trophy's rate rises toward a ceiling as the player base matures, so the program fits a saturating curve in logit space — logit(rate) = L_inf − A·(1+t)^(−γ) — over four decay rates, averaged by fit quality. Story trophies are kept in chapter order, a trophy nobody has earned yet borrows a tenth of its tier's forecast, and a frozen page (every read identical) falls back to a refresh prior: the tracked-player count jumps while earners creep, so rates fall. The spread carries a refresh allowance plus a random-walk term that grows with the horizon. Recalibrated Nov 2026: the frozen prior's median is shifted down 0.30 logit and both branches' bands are shrunk (frozen ×0.65, live ×0.70), measured on the reconstructed practice answers and a simulated final-regime week. Similarity-weighted pooling toward tier/chapter peer medians was tried and rejected — it worsened the simulated loss at every blend weight.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Reads each case at `$MAKEFUN_INPUT_DIR/*.in` and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR/*.out`. Standard library only, deterministic, no network.

## Validation

On the kit's two practice cases (against the answer reconstructed from the rules' own refresh description): pinball loss 0.13–0.17, coverage_90 0.98, typical miss 0.01–0.03 points — against the flat baseline's 1.35–1.69 loss, 0.00 coverage, 0.67–0.73 miss. On a simulated final-regime week (live reads cut at day 2.5, target day 7): loss 0.37–0.40, coverage_90 0.87, typical miss 2.8–4.2 points, against the flat baseline's 1.15 and the trend baseline's 1.21–1.22. (Practice coverage is capped at 0.98 by construction: the single miss is the pinned 100% trophy, whose 95% quantile can never reach exactly 100.)

## make.fun

Built for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250) on make.fun.

Optional: earn referral rewards — [join make.fun with my referral link](https://make.fun/r/s7sxc3qn).
