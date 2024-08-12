# Litecoin Growth Simulation

This repository contains a Python library for simulating user growth and daily transactions in a system, influenced by various factors including logistic growth models and external factors like BTC hashrate. The library also includes Monte Carlo simulations to account for uncertainty and confidence intervals.

---
## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Simulate User Growth and Transactions](#simulate-user-growth-and-transactions)
    - [Simulate Growth with BTC Hashrate](#simulate-growth-with-btc-hashrate)
6. [Monte Carlo Simulation](#monte-carlo-simulation)
    - [Monte Carlo Simulation with BTC Hashrate Influence](#monte-carlo-simulation-with-btc-hashrate-influence)
7. [Parameters](#parameters)
8. [Contributors/Maintainers](#contributorsmaintainers)
9. [Contributing](#contributing)
10. [License](#license)
---

## Directory Structure

```bash
/project_root
|-- /assets
|   |-- growth_and_transactions.png
|   |-- growth_and_transactions.png
|   |-- growth_and_transactions_CI.png
|   |-- growth_and_transactions_CI_hashrate.png
|
|-- /white_paper
|   |-- Modeling Litecoin User Growth...Shpaner,Leonid.pdf
|
|-- LICENSE.MD
|-- README.md
|-- main.py
|-- requirements.txt
```

## Features

- **Simulate User Growth and Transactions**: Model user growth over time and calculate daily transactions.
- **Simulate Growth with BTC Hashrate Influence**: Integrate external factors like BTC hashrate to simulate growth more realistically.
- **Monte Carlo Simulations**: Perform multiple simulations to generate confidence intervals and understand the variability in predictions.
- **Customizable Plots**: Generate and save plots for analysis, with support for various customization options.

## Installation

Clone the repository

```bash
git clone https://github.com/litecoin-foundation/litecoin_analytics.git
pip install -r requirements.txt
```

## Usage 

For now, you can rely on the [`main.py`](https://github.com/litecoin-foundation/litecoin_analytics/blob/main/main.py) script to access and utilize the key functions of the project. The script includes all the essential functionalities that will eventually be packaged into a standalone library.


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


## Contributors/Maintainers

<img align="left" width="150" height="150" src="https://www.leonshpaner.com/author/leon-shpaner/avatar_hu48de79c369d5f7d4ff8056a297b2c4c5_1681850_270x270_fill_q90_lanczos_center.jpg">

[Leonid Shpaner](https://github.com/lshpaner) is a Data Scientist at UCLA Health. With over a decade experience in analytics and teaching, he has collaborated on a wide variety of projects within financial services, education, personal development, and healthcare. He serves as a course facilitator for Data Analytics and Applied Statistics at Cornell University and is a lecturer of Statistics in Python for the University of San Diego's M.S. Applied Artificial Intelligence program.  

<br>
<br>
<br>

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

## License

This project is licensed under the [MIT License](https://github.com/litecoin-foundation/litecoin_analytics/blob/main/LICENSE.MD).
