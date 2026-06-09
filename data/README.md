# Data — not redistributed

This analysis consumes two external datasets. Neither is redistributed in this
repository, for licensing reasons. Download each from its upstream source and
place it in this folder (or adjust `DATA_PATH` / `ACTIVE_CSV` in the notebook).

## 1. Spock retweet cascade — `example_book.csv`

219 events with `(time_seconds, magnitude)`, where `magnitude` is the
retweeting user's follower count; the first row (`time = 0`) is the seed event.
This is the NYT "Spock" retweet cascade discussed in Rizoiu et al. (2017).

* Source: <https://github.com/s-mishra/featuredriven-hawkes/blob/master/code/example_book.csv>
* Upstream license: **GNU GPL v3**
* Cite: Mishra, Rizoiu & Xie. *Feature Driven and Point Process Approaches for
  Popularity Prediction.* CIKM 2016. doi:10.1145/2983323.2983812

Used by `notebooks/main.ipynb` (real-data fit, Phase 0 / 1, and the
Hardiman–Bouchaud cross-check).

## 2. ACTIVE dataset — `data.csv`

~23,000 retweet cascades (`time,magnitude`), where `magnitude` is the follower
count; rows with `time == 0` mark cascade boundaries.

* Source: Rizoiu group cascade-influence repository (ACTIVE dataset).
* Upstream license: **CC BY-NC 4.0**

Used by `notebooks/main.ipynb` (the ACTIVE faithful-constraint section).

## Why these are not included

The Spock CSV is GPLv3 upstream and would impose copyleft obligations
incompatible with this repository's MIT license if redistributed here. The
ACTIVE dataset is CC BY-NC 4.0 (non-commercial). The underlying retweet data
may also carry platform/user rights independent of the repository license.
Keeping the data external lets the MIT license apply cleanly to all
distributed content while preserving full reproducibility.
