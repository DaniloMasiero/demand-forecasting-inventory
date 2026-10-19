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

## First look at the data
- 1,067,371 invoice lines, 2009-12-01 to 2011-12-09 (104 weeks), 92% from the UK.
- Removed: 19.5k cancellation lines (invoice starting with "C"), lines with quantity or price <= 0,
  non-product codes (POST, DOT, M, BANK CHARGES, AMAZONFEE...) and 34k exact duplicate lines.
- 4,878 products remain. 300 cover 50% of units and 1,052 cover 80% (classic Pareto);
  543 sell in at least 90 of the 104 weeks, so they have a usable history.
- Strong Q4 seasonality: the peak week sells 2.5x the median week. With only two years there is
  just one past season to learn from, which limits yearly-seasonality models.

## Plan
1. Clean data: cancellations, returns, non-product codes, outliers.
2. Aggregate to SKU x week; pick the top SKUs by volume (Pareto).
3. Baselines: naive (last week), moving average, seasonal naive.
4. Model: LightGBM with lag, rolling and calendar features, one global model for all SKUs.
5. Validation: time-based backtesting (rolling origin), never random splits.
6. Metrics: WAPE and bias, plus the cost of error (overstock vs. stockout).
7. Recommendation: where the model helps, where the baseline is good enough.

## Results
_To be filled in._

## What I would do differently
_To be filled in._
