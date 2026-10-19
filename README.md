# Weekly Demand Forecasting for Inventory Planning

> Status: in progress

## Business problem
A retailer has to decide how much of each product to stock every week. Overstock ties up cash
and risks expiry; understock loses sales. Goal: forecast next weeks' demand per product (SKU)
and show how much better than the current rule of thumb (last weeks' average) it is, in units
and in money.

## Data
UCI Online Retail II: real transactions of a UK online retailer, Dec 2009 – Dec 2011
(~1M invoice lines). Not committed to the repo (`data/` is git-ignored).

## Data cleaning (`src/prepare.py`)
Raw: 1,067,371 invoice lines, Dec 2009 to Dec 2011, 92% from the UK.

| Step | Lines removed | Why |
|---|---|---|
| Exact duplicates | 34,335 | Same invoice, SKU, quantity and timestamp twice: a system artifact, not demand |
| Sales cancelled later | 6,332 | A "C" invoice cancels an earlier sale (same customer, SKU, quantity). Dropping only the "C" line would keep phantom demand, e.g. 80,995 units ordered and cancelled 12 minutes later |
| Quantity or price <= 0 | | Stock adjustments (damages, missing) and free samples |
| Non-product codes | | POST, DOT, M, BANK CHARGES, AMAZONFEE... are fees and manual entries |

997,035 clean lines remain (93.4%). They are aggregated to units per SKU per week (Monday to
Sunday). The first and last weeks are partial and dropped, leaving 104 weeks. Weeks with no
sales become 0, which is real information for a forecast.

**Which SKUs:** only the 526 that sold in at least 85% of weeks (of 4,669). They cover 47% of
units. Short-lived products would need a different method (new-product or intermittent-demand
forecasting), so they are out of scope here.

## Validation and metrics (`src/backtest.py`)
- **Rolling-origin backtest:** 16 origins, one per week, Jul to Nov 2011. At each origin the
  forecaster sees only the past and predicts the next 4 weeks (a typical replenishment lead
  time). A random train/test split would leak the future into training.
- **WAPE** (total absolute error / total demand) instead of MAPE: MAPE explodes on low-volume
  weeks and weights a 1-unit SKU like a 1,000-unit one.
- **Bias** (total forecast minus demand, over demand): WAPE hides direction, and inventory
  cost is not symmetric.

## Baselines (`src/baselines.py`)
| Baseline | WAPE (SKU-week) | Bias | WAPE (total-week) |
|---|---|---|---|
| Moving average 12 weeks | **0.622** | -9.8% | **0.127** |
| Moving average 4 weeks | 0.659 | -6.6% | 0.132 |
| Naive (last week) | 0.779 | -4.3% | 0.159 |
| Seasonal naive, scaled to current level | 0.942 | +19.4% | 0.226 |
| Seasonal naive (same week last year) | 0.994 | +34.0% | 0.345 |

Error barely changes from 1 to 4 weeks ahead.

**What this says:**
1. **Demand per SKU is very noisy.** The median SKU has a weekly coefficient of variation of
   1.1, typical of a wholesaler with large, irregular orders. Error drops from 62% per SKU
   to 13% for total demand. The noise is in individual orders, not in the overall trend.
   For planning, that points to safety stock per SKU, or to forecasting at a higher level
   and allocating down.
2. **Last year is a bad guide for these SKUs.** The same products sold a median of 30% fewer
   units in Aug-Nov 2011 than in Aug-Nov 2010, as they aged in their life cycle. Seasonal naive
   repeats last year and over-forecasts by 34%; scaling to the current level halves the bias
   but still loses to a plain moving average.
3. **Benchmark to beat: 12-week moving average, WAPE 0.622, bias -9.8%.** Its under-forecast
   comes from the Q4 ramp-up, which a backward-looking average reacts to late. A model has
   to beat it on WAPE without making the bias worse.

## Next steps
1. Model: LightGBM with lag, rolling and calendar features, one global model for all SKUs.
2. Same backtest and metrics as the baselines, plus the cost of error (overstock vs. stockout).
3. Recommendation: where the model helps, where the baseline is good enough.

## What I would do differently
_To be filled in._

## How to run
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# download Online Retail II from the UCI repository into data/raw/
python src/load_data.py && python src/prepare.py && python src/baselines.py
```
