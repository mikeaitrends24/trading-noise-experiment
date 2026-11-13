# Trading Noise Experiment

A Python experiment exploring how an EMA crossover optimizer can find profitable-looking strategies in synthetic random price data.

## What This Project Tests

This project examines a simple but important backtesting problem:

Can an optimizer find apparently profitable trading strategies even when the underlying price series contains no genuine market signal?

The experiment uses synthetic price series generated from Gaussian random returns.

## Experiment Setup

- Synthetic four-hour price series
- 8,760 return bars per series
- EMA crossover strategy
- 178 fast/slow parameter combinations
- Profit-factor-based optimization
- Separate permutation experiment using 200 shuffles per series

## Files

- `permutation_demo.py` — Python experiment script
- `results.txt` — Actual execution output, when available

## How to Run

Clone or download this repository.

Install the dependencies required by the script.

Run:

```bash
python permutation_demo.py
```

Check the script for its specific configuration and dependency requirements before running.

## Important Limitations

- The price data is synthetic, not historical market data.
- The experiment does not establish a tradable strategy.
- Transaction costs and slippage must be considered.
- A profit factor threshold of 1.05 is a reporting threshold used in this experiment, not a validated definition of an edge.
- Walk-forward validation has not yet been completed.
- Results depend on the random seeds and experiment configuration.

## Reproducibility

The script and its configuration are provided for inspection. Numerical results should be checked against the actual output produced by running the script.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice or a recommendation to trade.
