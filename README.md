# Weekly Demand Forecasting for Inventory Planning

> Status: complete

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

## Model: global LightGBM (`src/model.py`)
- **One model for all 526 SKUs**, not one per SKU. Two noisy years per product is too little to
  fit a model each (ARIMA per SKU would overfit). Together, the products share patterns such as
  the Q4 ramp-up and life-cycle decline.
- **Scale-free target:** units divided by the SKU's 12-week mean, so a 10-unit and a 1,000-unit
  SKU look alike. Sample weights equal to that mean bring the loss back to units.
- **Features known at the origin:** recent lags, 4- and 26-week means, volatility, share of zero
  weeks, short-term trend, last year's level around the target week, weeks to Christmas,
  week of year and horizon.
- **Retrained at every origin** with only targets already observed at that point.

### Choosing the variant on a validation period, not on the test
Every variant was compared on 16 earlier origins (Mar to Jun 2011). The test period was used
once, for the chosen variant. Rule: lowest WAPE among variants with |bias| <= 10%.

| Variant (validation) | WAPE | Bias |
|---|---|---|
| L1 loss | 0.627 | -20.3% |
| L1 blended 50/50 with MA12 | 0.629 | -11.6% |
| Quantile 0.55 | 0.638 | -13.1% |
| **Quantile 0.60 (chosen)** | **0.655** | **-5.2%** |
| Moving average 12w (baseline) | 0.660 | -3.0% |
| L2 loss (mean), with or without week of year | 0.692-0.694 | +5-6% |
| L1 + rescaling bias correction, with or without week of year | 0.716-0.718 | +10% |

The key finding: **L1 loss learns the median**, and demand is right-skewed (big irregular
orders), so the median sits below the mean. The L1 model "wins" on WAPE by under-forecasting
by 20%, which means stockouts. Rescaling the forecast overshoots. Training on a quantile
slightly above the median (0.60) gives the trade-off directly.

### Test results (Jul to Nov 2011, 16 origins, 4 weeks ahead)
| | WAPE | Bias |
|---|---|---|
| **LightGBM, quantile 0.60** | **0.593** | -10.3% |
| Moving average 12w | 0.622 | -9.8% |

- 4.7% lower error than the best baseline with the same bias. The model is better on 58% of SKUs.
- In the Oct-Nov peak it wins on both: WAPE 0.600 vs 0.621, bias -10.3% vs -13.6%. The moving
  average reacts late to the ramp-up; the model sees it coming through the calendar features.
- In Jul-Sep it has lower error, but it under-forecasts more (-10.4% vs -5.8%).
- Week of year carries 38% of the gain. With a single past season this is a risk: the model may
  be learning 2010's calendar. Dropping it did not help on validation, but more years of data
  would be needed to trust it.

## What the better forecast is worth (`src/inventory.py`)
A buyer orders forecast + safety stock to cover the 4-week lead time. A better forecast needs
less safety stock for the same protection. No cost assumptions are needed: safety stock is
valued at each SKU's median 2011 selling price.

**Textbook sizing fails out of sample.** Safety stock = mean error + 1.645 x std of error (aims
at 95% of cycles without stockout), sized on the validation period, reaches only 80% (moving
average) and 84% (model) on the test period. Errors in the Q4 peak are larger than in spring,
and lumpy demand has fatter tails than the normal curve assumes.

**Same service level, less inventory.** Sweeping the safety factor shows how much safety stock
each forecast needs to reach a given share of cycles without stockout on the test period:

| Service level | Moving average 12w | LightGBM | Saved |
|---|---|---|---|
| 80% | GBP 269k | GBP 230k | **GBP 39k (14.5%)** |
| 85% | GBP 359k | GBP 303k | **GBP 56k (15.6%)** |
| 90% | GBP 498k | GBP 417k | **GBP 81k (16.2%)** |
| 95% | GBP 808k | GBP 713k | **GBP 95k (11.8%)** |

A 4.7% lower forecast error turns into 12-16% less safety stock: the model's errors are not just
smaller on average but less dispersed, and safety stock is driven by the dispersion.
Caveat: this curve is read on the test period itself, so it compares the two forecasts fairly
but is not a sizing rule to deploy as is.

## Recommendation
Use the model for the 526 steady SKUs, with safety stock sized per SKU from its backtest errors
and re-calibrated before Q4, when errors grow. At an 85-90% service level it frees about
GBP 56-81k of safety stock on these products alone. The gain is largest in the Q4 peak, when
the moving average falls furthest behind.

## Next steps
1. Size safety stock with empirical error quantiles per season instead of the normal formula.
2. Forecast at a higher level (category or total) and allocate down; compare.
3. A separate method for the other 4,100 short-lived SKUs (intermittent demand).

## What I would do differently
- With more history, validate seasonality across several years before trusting week-of-year.
- Define the bias rule from the business costs up front, instead of a fixed 10% threshold.

## How to run
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# download Online Retail II from the UCI repository into data/raw/
python src/load_data.py && python src/prepare.py && python src/baselines.py
python src/validate.py && python src/validate.py --round 2   # model choice
python src/model.py                                          # final test
python src/inventory.py                                      # safety stock value
```
