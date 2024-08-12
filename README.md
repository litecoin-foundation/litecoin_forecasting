# Litecoin Growth Simulation

This repository contains a Python library for simulating user growth and daily transactions in a system, influenced by various factors including logistic growth models and external factors like BTC hashrate. The library also includes Monte Carlo simulations to account for uncertainty and confidence intervals.

## Features

- **Simulate User Growth and Transactions**: Model user growth over time and calculate daily transactions.
- **Simulate Growth with BTC Hashrate Influence**: Integrate external factors like BTC hashrate to simulate growth more realistically.
- **Monte Carlo Simulations**: Perform multiple simulations to generate confidence intervals and understand the variability in predictions.
- **Customizable Plots**: Generate and save plots for analysis, with support for various customization options.

## Installation

To use this library, you need Python 3.x installed. You can install the required libraries using `pip`:

```bash
pip install numpy pandas requests matplotlib
```

## Simulate User Growth and Transactions

```python
results = simulate_user_growth_and_transactions(
    initial_users=1000,
    carrying_capacity=100000,
    growth_rate=0.10,
    usage_rate=0.8,
    time_steps=365,
    start_date="2024-01-01",
    figsize=(12, 6),
    plot=True,
    return_results=True,
    label_fontsize=12,
    tick_fontsize=10,
    save_plot=True,
    image_path_png="user_growth.png",
    image_path_svg="user_growth.svg"
)
```

![](https://drive.google.com/uc?export=view&id=1-0Nn_AGfnoYHKgSImkdSuHW_VtLtqKQJ)

### Simulate Growth with BTC Hashrate

```python
results = simulate_growth_with_btc_hashrate(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    usage_rate=0.8,
    days=365,
    figsize=(12, 6),
    plot=True,
    return_results=True,
    label_fontsize=12,
    tick_fontsize=10,
    save_plot=True,
    image_path_png="growth_btc_hashrate.png",
    image_path_svg="growth_btc_hashrate.svg"
)
```

![](https://drive.google.com/uc?export=view&id=1FNqnVfpZ3ghvngYA_lOEfqExAEPgvjdy)


## Monte Carlo Simulation

```python
results = simulate_monte_carlo_growth(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    time_steps=365,
    num_simulations=1000,
    confidence_levels=[95, 99],
    x_axis="days",
    start_date="2024-01-01",
    plot=True,
    save_plot=True,
    image_path_png="monte_carlo_growth.png",
    image_path_svg="monte_carlo_growth.svg"
)
```

![](https://drive.google.com/uc?export=view&id=1-9os5P6T4MlExUitSSRvrXJ2oaEUlD30)

### Monte Carlo Simulation with BTC Hashrate Influence

```python
results = simulate_monte_carlo_growth_with_hashrate(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    time_steps=365,
    num_simulations=1000,
    confidence_levels=[95, 99],
    x_axis="days",
    start_date="2024-01-01",
    days=365,
    plot=True,
    save_plot=True,
    image_path_png="monte_carlo_hashrate.png",
    image_path_svg="monte_carlo_hashrate.svg"
)
```

![](https://drive.google.com/uc?export=view&id=1-BArBGaAgavVPel1U3iX-XZukXSjX_r4)



## Parameters

All functions in the library provide flexibility through several parameters. These include:

- `initial_users`: Initial number of users.  
- `carrying_capacity`: Maximum number of users the system can support.  
- `growth_rate`: Growth rate per time step.  
- `usage_rate`: Proportion of users who use the system daily.  
- `time_steps`: Number of days to simulate.  
- `plot`: Whether to plot the results.  
- `return_results`: Whether to return the results as a DataFrame.  
- `save_plot`: Whether to save the generated plots.

Refer to the docstrings within each function for detailed descriptions of all parameters.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.
