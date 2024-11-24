################################################################################
########################## Import Requisite Libraries ##########################
################################################################################

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


################################################################################
##################### Simulate User Growth and Transactions ####################
################################################################################


def simulate_user_growth_and_transactions(
    initial_users=1000,
    carrying_capacity=100000,
    growth_rate=0.10,
    usage_rate=0.8,
    time_steps=365,
    start_date=None,
    figsize=(12, 6),
    plot=True,
    return_results=False,
    label_fontsize=12,
    tick_fontsize=10,
    image_path_png=None,  # Path to save PNG images
    image_path_svg=None,  # Path to save SVG images
    save_plot=False,  # Whether to save the plot
    **kwargs,
):
    """
    Simulates user growth using a logistic growth model and calculates daily
    transactions.

    Parameters:
    -----------
    initial_users : int, optional (default=1000)
        Initial number of users.

    carrying_capacity : int, optional (default=100000)
        Maximum number of users the system can support.

    growth_rate : float, optional (default=0.10)
        Growth rate per time step.

    usage_rate : float, optional (default=0.8)
        Proportion of users who use the wallet daily.

    time_steps : int, optional (default=365)
        Number of time steps to simulate (e.g., days).

    start_date : str or None, optional (default=None)
        If provided, should be a string in 'YYYY-MM-DD' format.
        The simulation will use actual calendar dates starting from this date on
        the x-axis.

    figsize : tuple, optional (default=(12, 6))
        Size of the plot figure.

    plot : bool, optional (default=True)
        Whether to plot the results.

    return_results : bool, optional (default=False)
        Whether to return the results as a DataFrame.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    image_path_png : str, optional
        File path to save the plot as a PNG.

    image_path_svg : str, optional
        File path to save the plot as an SVG.

    save_plot : bool, optional (default=False)
        Whether to save the plot as an image.

    **kwargs : dict, optional
        Additional keyword arguments to pass to the plot function.

    Returns:
    --------
    results_df : pandas.DataFrame (optional)
        DataFrame containing the time (or dates), users, and daily transactions,
        returned if return_results=True.
    """

    # Raise error if save_plot is True but no image path is provided
    if save_plot and not (image_path_png or image_path_svg):
        raise ValueError(
            "To save plots, you must specify at least one of "
            "'image_path_png' or 'image_path_svg'."
        )

    def logistic_growth(t, initial, carrying_capacity, rate):
        return carrying_capacity / (
            1 + ((carrying_capacity - initial) / initial) * np.exp(-rate * t)
        )

    # Simulate the process
    time = np.arange(time_steps)
    users = logistic_growth(
        time,
        initial_users,
        carrying_capacity,
        growth_rate,
    )
    daily_transactions = users * usage_rate

    # Handle the x-axis as dates if start_date is provided
    if start_date:
        dates = pd.date_range(start=start_date, periods=time_steps)
        x_axis = dates
        x_label = "Date"
    else:
        x_axis = time
        x_label = "Days"

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(x_axis, users, label="Total Users", **kwargs)
        plt.plot(
            x_axis,
            daily_transactions,
            label="Daily Transactions",
            **kwargs,
        )
        plt.xlabel(x_label, fontsize=label_fontsize)
        plt.ylabel("Number of Users/Transactions", fontsize=label_fontsize)
        plt.legend()
        plt.grid(True)
        plt.title(
            "User Growth and Daily Transactions Over Time",
            fontsize=label_fontsize,
        )
        plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        # Save the plot if specified
        if save_plot:
            if image_path_png:
                plt.savefig(image_path_png, bbox_inches="tight", format="png")
            if image_path_svg:
                plt.savefig(image_path_svg, bbox_inches="tight", format="svg")

        plt.show()
        plt.close()

    if return_results:
        results_df = pd.DataFrame(
            {
                x_label: x_axis,
                "Total Users": users,
                "Daily Transactions": daily_transactions,
            }
        )
        return results_df


################################################################################
################## Simulate User Growth With BTC Transactions ##################
################################################################################


def simulate_growth_with_btc_hashrate(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    usage_rate=0.8,
    days=365,
    figsize=(12, 6),
    plot=True,
    return_results=False,
    label_fontsize=12,
    tick_fontsize=10,
    image_path_png=None,  # Path to save PNG images
    image_path_svg=None,  # Path to save SVG images
    save_plot=False,  # Whether to save the plot
    **kwargs,
):
    """
    Simulates user growth influenced by BTC hashrate data using a logistic
    growth model.

    Parameters:
    -----------
    initial_users : int, optional (default=1000)
        Initial number of users.

    carrying_capacity : int, optional (default=100000)
        Maximum number of users the system can support.

    base_growth_rate : float, optional (default=0.10)
        Base growth rate per time step.

    usage_rate : float, optional (default=0.8)
        Proportion of users who use the wallet daily.

    days : int, optional (default=365)
        Number of days to simulate.

    figsize : tuple, optional (default=(12, 6))
        Size of the plot figure.

    plot : bool, optional (default=True)
        Whether to plot the results.

    return_results : bool, optional (default=False)
        Whether to return the results as a DataFrame.

    label_fontsize : int, optional (default=12)
        Font size for axis labels.

    tick_fontsize : int, optional (default=10)
        Font size for axis tick labels.

    image_path_png : str, optional
        File path to save the plot as a PNG.

    image_path_svg : str, optional
        File path to save the plot as an SVG.

    save_plot : bool, optional (default=False)
        Whether to save the plot as an image.

    **kwargs : dict, optional
        Additional keyword arguments to pass to the plot function.

    Returns:
    --------
    results_df : pandas.DataFrame (optional)
        DataFrame containing the time, users, and daily transactions,
        returned if return_results=True.
    """

    # Check if save_plot is True but neither image path is provided
    if save_plot and not (image_path_png or image_path_svg):
        raise ValueError(
            "To save plots, you must specify at least one of "
            "'image_path_png' or 'image_path_svg'."
        )

    # Fetch BTC hashrate data from CoinGecko API
    response = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": str(days)},
    )
    data = response.json()

    # Extract and normalize the hashrate values (here we use prices as an example;
    # replace with hashrate if available)
    prices = np.array([item[1] for item in data["prices"]])
    btc_hashrate_normalized = (prices - np.min(prices)) / (
        np.max(prices) - np.min(prices)
    )

    # Adjust growth rate based on actual BTC hashrate
    growth_rate = base_growth_rate * btc_hashrate_normalized

    # Logistic growth function
    def logistic_growth(t, initial, carrying_capacity, rate):
        return carrying_capacity / (
            1 + ((carrying_capacity - initial) / initial) * np.exp(-rate * t)
        )

    # Simulate the process
    time_steps = len(btc_hashrate_normalized)
    time = np.arange(time_steps)
    users = logistic_growth(
        t=time,
        initial=initial_users,
        carrying_capacity=carrying_capacity,
        rate=growth_rate,
    )
    daily_transactions = (
        users * usage_rate
    )  # assuming that a percentage of users use the wallet daily

    # Plot the results
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(time, users, label="Total Users", **kwargs)
        plt.plot(time, daily_transactions, label="Daily Transactions", **kwargs)
        plt.xlabel("Days", fontsize=label_fontsize)
        plt.ylabel("Number of Users/Transactions", fontsize=label_fontsize)
        plt.title(
            "User Growth and Daily Transactions Simulation\n"
            "with BTC Hashrate Influence",
            fontsize=label_fontsize,
        )
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        # Save the plot if specified
        if save_plot:
            if image_path_png:
                plt.savefig(image_path_png, bbox_inches="tight", format="png")
            if image_path_svg:
                plt.savefig(image_path_svg, bbox_inches="tight", format="svg")

        plt.show()
        plt.close()

    # Return the results in a DataFrame
    if return_results:
        results_df = pd.DataFrame(
            {
                "Days": time,
                "Total Users": users,
                "Daily Transactions": daily_transactions,
            }
        )
        return results_df


################################################################################
############################ Monte Carlo Simulation ############################
################################################################################


def simulate_monte_carlo_growth(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    time_steps=365,
    num_simulations=1000,
    confidence_levels=[95, 99],
    confidence_colors=["lightblue", "lightgreen"],
    x_axis="days",  # Options: "days", "quarters", "datetime"
    start_date=None,  # Required if x_axis is "datetime"
    figsize=(12, 6),
    plot=True,
    return_results=False,
    title="LTC Growth Rate Simulation with Confidence Intervals",
    xlabel="Days",
    ylabel="Number of Users",
    label_fontsize=14,
    tick_fontsize=12,
    save_plot=False,
    image_path_png=None,  # Path to save PNG images
    image_path_svg=None,  # Path to save SVG images
    save_filename="monte_carlo_growth",
    bbox_inches="tight",
    plot_color="blue",
    plot_linestyle="-",
    fill_between_alpha=0.3,
    **kwargs,
):
    def logistic_growth(t, initial, carrying_capacity, rate):
        return carrying_capacity / (
            1 + ((carrying_capacity - initial) / initial) * np.exp(-rate * t)
        )

    try:
        # Convert single values to lists for consistency
        if isinstance(confidence_levels, (float, int)):
            confidence_levels = [confidence_levels]
        if isinstance(confidence_colors, str):
            confidence_colors = [confidence_colors]

        if len(confidence_colors) != len(confidence_levels):
            raise ValueError(
                f"The number of confidence colors must match the "
                f"number of confidence levels."
            )

        # Set up the x-axis based on user input
        if x_axis == "days":
            time = np.arange(time_steps)
            xlabel = "Days"
        elif x_axis == "quarters":
            # Calculate time steps and ticks for quarters
            days_in_quarter = time_steps // 4
            quarters = ["Q1", "Q2", "Q3", "Q4"]
            quarter_ticks = np.linspace(0, time_steps, num=5)[:-1]
            time = np.arange(time_steps)
            xlabel = "Quarters"
        elif x_axis == "datetime":
            if start_date is None:
                raise ValueError(
                    f"`start_date` must be provided when x_axis is set "
                    f"to 'datetime'."
                )

            time = pd.date_range(start=start_date, periods=time_steps)
            xlabel = "Date"
        else:
            raise ValueError(
                "`x_axis` must be one of 'days', 'quarters', or 'datetime'."
            )

        # Raise error if save_plot is True but no image path is provided
        if save_plot and not (image_path_png or image_path_svg):
            raise ValueError(
                "To save plots, you must specify at least one of "
                "'image_path_png' or 'image_path_svg'."
            )

        # Simulate multiple runs
        all_users = []

        for _ in range(num_simulations):
            # Add randomness to the growth rate
            random_growth_rate = base_growth_rate + np.random.normal(
                0, 0.01, size=time_steps
            )

            # Compute the logistic growth for this simulation
            users = logistic_growth(
                t=np.arange(time_steps),
                initial=initial_users,
                carrying_capacity=carrying_capacity,
                rate=random_growth_rate,
            )
            all_users.append(users)

        all_users = np.array(all_users)

        # Calculate percentiles for confidence intervals
        confidence_intervals = {}
        for confidence_level in confidence_levels:
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = np.percentile(all_users, lower_percentile, axis=0)
            upper_bound = np.percentile(all_users, upper_percentile, axis=0)
            confidence_intervals[confidence_level] = (lower_bound, upper_bound)

        # Calculate the median prediction
        median_prediction = np.median(all_users, axis=0)

        # Prepare DataFrame with results
        results_df = pd.DataFrame(
            {"Time": time, "Median Prediction": median_prediction}
        )
        for confidence_level, (
            lower_bound,
            upper_bound,
        ) in confidence_intervals.items():
            results_df[f"Lower {confidence_level}% CI"] = lower_bound
            results_df[f"Upper {confidence_level}% CI"] = upper_bound

        if plot:
            plt.figure(figsize=figsize)

            # Plot the median prediction
            plt.plot(
                time,
                median_prediction,
                color=plot_color,
                linestyle=plot_linestyle,
                label="Median Prediction",
                **kwargs,
            )

            # Plot confidence intervals
            for idx, (confidence_level, (lower_bound, upper_bound)) in enumerate(
                confidence_intervals.items()
            ):
                plt.fill_between(
                    time,
                    lower_bound,
                    upper_bound,
                    alpha=fill_between_alpha,
                    color=confidence_colors[idx],
                    label=f"{confidence_level}% CI",
                    **kwargs,
                )

            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.title(title, fontsize=label_fontsize)

            # Set the x-axis ticks and labels for quarters
            if x_axis == "quarters":
                plt.xticks(
                    ticks=quarter_ticks,
                    labels=quarters,
                    fontsize=tick_fontsize,
                )
            else:
                plt.xticks(fontsize=tick_fontsize)

            plt.yticks(fontsize=tick_fontsize)
            plt.legend(fontsize=label_fontsize)
            plt.grid(True)

            if save_plot:
                if image_path_png:
                    plt.savefig(
                        f"{image_path_png}/{save_filename}.png",
                        bbox_inches=bbox_inches,
                    )
                if image_path_svg:
                    plt.savefig(
                        f"{image_path_svg}/{save_filename}.svg",
                        bbox_inches=bbox_inches,
                    )

            plt.show()

        if return_results:
            return results_df

    except Exception as e:
        print(f"An error occurred: {e}")


################################################################################
########################### Monte Carlo With Hashrate ##########################
################################################################################


def simulate_monte_carlo_growth_with_hashrate(
    initial_users=1000,
    carrying_capacity=100000,
    base_growth_rate=0.10,
    time_steps=365,
    num_simulations=1000,
    confidence_levels=[95, 99],
    confidence_colors=["lightblue", "lightgreen"],
    x_axis="days",  # Options: "days", "quarters", "datetime"
    start_date=None,  # Required if x_axis is "datetime"
    days=365,  # Number of days to fetch hashrate data for
    figsize=(12, 6),
    plot=True,
    return_results=False,
    title="LTC Growth Rate Simulation with Confidence Intervals",
    xlabel="Days",
    ylabel="Number of Users",
    label_fontsize=14,
    tick_fontsize=12,
    save_plot=False,
    image_path_png=None,  # Path to save PNG images
    image_path_svg=None,  # Path to save SVG images
    save_filename="monte_carlo_growth",
    bbox_inches="tight",
    plot_color="blue",
    plot_linestyle="-",
    fill_between_alpha=0.3,
    random_state=None,  # Add random_state parameter for reproducibility
    **kwargs,
):
    def fetch_and_normalize_hashrate(days=365):
        """
        Fetches BTC hashrate (or price) data from CoinGecko API and normalizes it.

        Parameters:
        -----------
        days : int, optional (default=365)
            Number of days to fetch data for.

        Returns:
        --------
        btc_hashrate_normalized : np.array
            Normalized hashrate (or price) data for the specified period.
        """
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": str(days)},
        )

        # Check if the response was successful
        if response.status_code != 200:
            raise ValueError(
                f"Error fetching data from CoinGecko API: {response.status_code}"
            )

        data = response.json()

        # Check if 'prices' is in the response data
        if "prices" not in data:
            raise KeyError("'prices' not found in response data from CoinGecko")

        # Extract and normalize the hashrate values (using prices as a proxy here)
        prices = np.array([item[1] for item in data["prices"]])
        btc_hashrate_normalized = (prices - np.min(prices)) / (
            np.max(prices) - np.min(prices)
        )

        return btc_hashrate_normalized

    def logistic_growth(t, initial, carrying_capacity, rate):
        return carrying_capacity / (
            1 + ((carrying_capacity - initial) / initial) * np.exp(-rate * t)
        )

    try:
        # Set the random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Fetch and normalize BTC hashrate data
        btc_hashrate_data = fetch_and_normalize_hashrate(days=days)

        # Adjust the time_steps to match the hashrate data length
        if len(btc_hashrate_data) != time_steps:
            time_steps = len(btc_hashrate_data)

        # Set up the x-axis based on user input
        if x_axis == "days":
            time = np.arange(time_steps)
            xlabel = "Days"
        elif x_axis == "quarters":
            # Calculate time steps and ticks for quarters
            days_in_quarter = time_steps // 4
            quarters = ["Q1", "Q2", "Q3", "Q4"]
            quarter_ticks = np.linspace(0, time_steps, num=5)[:-1]
            time = np.arange(time_steps)
            xlabel = "Quarters"
        elif x_axis == "datetime":
            if start_date is None:
                raise ValueError(
                    f"`start_date` must be provided when x_axis is set "
                    f"to 'datetime'."
                )
            time = pd.date_range(start=start_date, periods=time_steps)
            xlabel = "Date"
        else:
            raise ValueError(
                "`x_axis` must be one of 'days', 'quarters', or 'datetime'."
            )

        # Normalize and process hashrate data
        btc_hashrate_for_simulation = btc_hashrate_data[
            :time_steps
        ]  # Ensure matching length

        # Convert single values to lists for consistency
        if isinstance(confidence_levels, (float, int)):
            confidence_levels = [confidence_levels]
        if isinstance(confidence_colors, str):
            confidence_colors = [confidence_colors]

        if len(confidence_colors) != len(confidence_levels):
            raise ValueError(
                f"The number of confidence colors must match the "
                f"number of confidence levels."
            )

        # Raise error if save_plot is True but no image path is provided
        if save_plot and not (image_path_png or image_path_svg):
            raise ValueError(
                f"To save plots, you must specify at least one of "
                f"'image_path_png' or 'image_path_svg'."
            )

        # Simulate multiple runs
        all_users = []

        for _ in range(num_simulations):
            # Modify the growth rate based on actual hashrates per quarter
            random_growth_rate = (
                base_growth_rate * btc_hashrate_for_simulation
                + np.random.normal(0, 0.01, size=time.shape)
            )

            # Compute the logistic growth for this simulation
            users = logistic_growth(
                t=np.arange(time_steps),
                initial=initial_users,
                carrying_capacity=carrying_capacity,
                rate=random_growth_rate,
            )
            all_users.append(users)

        all_users = np.array(all_users)

        # Calculate percentiles for confidence intervals
        confidence_intervals = {}
        for confidence_level in confidence_levels:
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = np.percentile(all_users, lower_percentile, axis=0)
            upper_bound = np.percentile(all_users, upper_percentile, axis=0)
            confidence_intervals[confidence_level] = (lower_bound, upper_bound)

        # Calculate the median prediction
        median_prediction = np.median(all_users, axis=0)

        # Prepare a DataFrame for correlation calculation
        correlation_df = pd.DataFrame(
            {
                "Hashrate": btc_hashrate_for_simulation,
                "MedianUserGrowth": median_prediction,
            }
        )

        # Calculate correlation using pandas .corr() method
        correlation = correlation_df.corr().loc["Hashrate", "MedianUserGrowth"]
        print(
            f"Correlation between hash rate and median user growth: {correlation:.4f}"
        )

        # Prepare DataFrame with results
        results_df = pd.DataFrame(
            {"Time": time, "Median Prediction": median_prediction}
        )
        for confidence_level, (
            lower_bound,
            upper_bound,
        ) in confidence_intervals.items():
            results_df[f"Lower {confidence_level}% CI"] = lower_bound
            results_df[f"Upper {confidence_level}% CI"] = upper_bound

        if plot:
            plt.figure(figsize=figsize)

            # Plot the median prediction
            plt.plot(
                time,
                median_prediction,
                color=plot_color,
                linestyle=plot_linestyle,
                label="Median Prediction",
                **kwargs,
            )

            # Plot confidence intervals
            for idx, (confidence_level, (lower_bound, upper_bound)) in enumerate(
                confidence_intervals.items()
            ):
                plt.fill_between(
                    time,
                    lower_bound,
                    upper_bound,
                    alpha=fill_between_alpha,
                    color=confidence_colors[idx],
                    label=f"{confidence_level}% CI",
                    **kwargs,
                )

            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.title(title, fontsize=label_fontsize)

            # Set the x-axis ticks and labels for quarters
            if x_axis == "quarters":
                plt.xticks(
                    ticks=quarter_ticks,
                    labels=quarters,
                    fontsize=tick_fontsize,
                )
            else:
                plt.xticks(fontsize=tick_fontsize)

            plt.yticks(fontsize=tick_fontsize)
            plt.legend(fontsize=label_fontsize)
            plt.grid(True)

            if save_plot:
                if image_path_png:
                    plt.savefig(
                        f"{image_path_png}/{save_filename}.png",
                        bbox_inches=bbox_inches,
                    )
                if image_path_svg:
                    plt.savefig(
                        f"{image_path_svg}/{save_filename}.svg",
                        bbox_inches=bbox_inches,
                    )

            plt.show()

        if return_results:
            return results_df

    except Exception as e:
        print(f"An error occurred: {e}")


################################################################################
############## Correlating Actual LTC User Data with BTC Hashrate ##############
################################################################################


class CryptoCorrelation:
    def __init__(
        self,
        api_key,
        days=365,
    ):
        """
        CryptoCorrelation Class

        This class provides functionality to fetch, process, and analyze
        the correlation between Litecoin active addresses and Bitcoin
        hashrate. It retrieves data from the CoinMetrics and CoinGecko
        APIs, normalizes the data if specified, and calculates the
        correlation between the two metrics. The class can also generate
        time series and correlation plots, with options to save the plots
        as PNG or SVG files.

        Constructor:
        ------------
        __init__(self, api_key, days=365)
            Constructor to initialize the CryptoCorrelation class.

            Parameters:
            -----------
            api_key : str
                API key for accessing the CoinMetrics API.
            days : int, optional
                Number of days to fetch Bitcoin hashrate data, default is 365.

        Attributes:
        -----------
        api_key : str
            API key for accessing the CoinMetrics API.
        days : int, optional
            Number of days to fetch Bitcoin hashrate data, default is 365.

        Methods:
        --------
            fetch_coin_active_addresses(coin="ltc", metric="AdrActCnt",
                                        start_date="2023-01-01",
                                        end_date="2024-01-01",
                                        normalize=False)
            Fetches Litecoin active addresses from the CoinMetrics API.

        fetch_bitcoin_hashrate(normalize=False)
            Fetches Bitcoin hashrate data from the CoinGecko API.

            connect_coin_users_to_btc_hashrate(start_date="2023-01-01",
                                        end_date="2024-01-01",
                                        normalize=False,
                                        plot_type="both",
                                        label_fontsize=14,
                                        tick_fontsize=12,
                                        textwrap_width=50,
                                        save_plot=False,
                                        image_path_png=None,
                                        image_path_svg=None,
                                        save_filename="crypto_correlation_plot",
                                        bbox_inches="tight")
            Connects Litecoin active addresses with Bitcoin hashrate,
            calculates correlation, and generates plots.
        """

        self.api_key = api_key
        self.days = days

    def fetch_coin_active_addresses(
        self,
        coin="ltc",
        metric="AdrActCnt",
        start_date="2023-01-01",
        end_date="2024-01-01",
        normalize=False,
    ):
        """
        Fetches Litecoin active addresses from the CoinMetrics API.

        Parameters:
        -----------
        coin : str, optional
            Cryptocurrency ticker, default is "ltc" for Litecoin.
        metric : str, optional
            Metric to fetch, default is "AdrActCnt" for active addresses.
        start_date : str, optional
            Start date for fetching data, default is "2023-01-01".
        end_date : str, optional
            End date for fetching data, default is "2024-01-01".
        normalize : bool, optional
            Whether to normalize the active addresses, default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing dates and active addresses.
        """
        url = (
            f"https://community-api.coinmetrics.io/v4/"
            f"timeseries/asset-metrics?assets={coin}&metrics={metric}&"
            f"start_time={start_date}&end_time={end_date}"
        )

        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            dates = [entry["time"] for entry in data["data"]]
            active_addresses = [float(entry[metric]) for entry in data["data"]]

            df = pd.DataFrame({"Date": dates, "Active Addresses": active_addresses})
            df["Date"] = pd.to_datetime(df["Date"])
            df["Date"] = df["Date"].dt.tz_localize(None)  # Ensure tz-naive datetime
            df.set_index("Date", inplace=True)

            if normalize:
                df["Active Addresses"] = (
                    df["Active Addresses"] - df["Active Addresses"].min()
                ) / (df["Active Addresses"].max() - df["Active Addresses"].min())

            return df
        else:
            raise ValueError(
                f"Error fetching data: {response.status_code} {response.text}"
            )

    def fetch_bitcoin_hashrate(self, normalize=False):
        """
        Fetches Bitcoin hashrate data from the CoinGecko API.

        Parameters:
        -----------
        normalize : bool, optional
            Whether to normalize the hashrate data, default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing dates and Bitcoin hashrate.
        """
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": str(self.days)},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error fetching Bitcoin data from API: {response.status_code}"
            )

        data = response.json()
        if "prices" not in data:
            raise KeyError(
                "'prices' not found in the response data from " "CoinGecko API"
            )

        hashrate = np.array([item[1] for item in data["prices"]])
        dates = pd.to_datetime([item[0] for item in data["prices"]], unit="ms")

        df = pd.DataFrame({"Date": dates, "Hashrate": hashrate})
        df["Date"] = df["Date"].dt.tz_localize(None)  # Ensure tz-naive datetime
        df.set_index("Date", inplace=True)

        if normalize:
            df["Hashrate"] = (df["Hashrate"] - df["Hashrate"].min()) / (
                df["Hashrate"].max() - df["Hashrate"].min()
            )

        return df

    def connect_coin_users_to_btc_hashrate(
        self,
        start_date="2023-01-01",
        end_date="2024-01-01",
        normalize=False,
        plot_type="both",
        label_fontsize=14,
        tick_fontsize=12,
        textwrap_width=50,
        save_plot=False,
        image_path_png=None,
        image_path_svg=None,
        save_filename="crypto_correlation_plot",
        bbox_inches="tight",
    ):
        """
        Connects Litecoin active addresses with Bitcoin hashrate and
        calculates correlation.

        Parameters:
        -----------
        start_date : str, optional
            Start date for fetching data, default is "2023-01-01".
        end_date : str, optional
            End date for fetching data, default is "2024-01-01".
        normalize : bool, optional
            Whether to normalize the data before calculating correlation and
            plotting, default is False.
        plot_type : str, optional
            Type of plot to generate: "time_series", "correlation", or "both",
            default is "both".
        label_fontsize : int, optional
            Font size for labels, default is 14.
        tick_fontsize : int, optional
            Font size for ticks, default is 12.
        textwrap_width : int, optional
            Width for wrapping the text in the title, default is 50 characters.
        save_plot : bool, optional
            Whether to save the plots as PNG or SVG, default is False.
        image_path_png : str, optional
            Directory path to save PNG images, default is None.
        image_path_svg : str, optional
            Directory path to save SVG images, default is None.
        save_filename : str, optional
            Base filename to use when saving the plots, default is
            "crypto_correlation_plot".
        bbox_inches : str, optional
            Bounding box to use when saving the figure, default is "tight".

        Returns:
        --------
        pd.DataFrame, float
            Combined DataFrame of Litecoin active addresses and Bitcoin hashrate,
            and the correlation coefficient between them.
        """
        # Validate plot_type
        valid_plot_types = ["time_series", "correlation", "both"]
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Invalid plot_type: {plot_type}. Must be one of "
                f"{valid_plot_types}."
            )

        # Check for save_plot requirement
        if save_plot and not (image_path_png or image_path_svg):
            raise ValueError(
                "To save plots, you must specify at least one of "
                "'image_path_png' or 'image_path_svg'."
            )

        # Fetch cryptocurrency's active addresses
        crypto_df = self.fetch_coin_active_addresses(
            start_date=start_date, end_date=end_date, normalize=normalize
        )

        # Fetch Bitcoin hashrate
        btc_df = self.fetch_bitcoin_hashrate(normalize=normalize)

        # Align the data by date
        combined_df = crypto_df.join(btc_df, how="inner")

        # Calculate correlation
        correlation = combined_df.corr().loc["Active Addresses", "Hashrate"]
        print(
            f"Correlation between Litecoin active addresses "
            f"and Bitcoin hashrate: {correlation:.4f}"
        )

        normalization_status = "Normalized" if normalize else "Non-Normalized"

        wrapped_title = textwrap.fill(
            f"Litecoin Active Addresses vs Bitcoin Hashrate "
            f"({normalization_status}, $r$ = {correlation:.4f})",
            width=textwrap_width,
        )

        wrapped_corr_title = textwrap.fill(
            f"Correlation between Bitcoin Hashrate and Litecoin "
            f"Active Addresses ({normalization_status}, $r$ = {correlation:.4f})",
            width=textwrap_width,
        )

        if plot_type in ["time_series", "both"]:
            plt.figure(figsize=(12, 6))
            plt.plot(
                combined_df.index,
                combined_df["Active Addresses"],
                label="Litecoin Active Addresses",
                color="blue",
            )
            plt.plot(
                combined_df.index,
                combined_df["Hashrate"],
                label="Bitcoin Hashrate",
                color="orange",
            )
            plt.xlabel("Date", fontsize=label_fontsize)
            plt.ylabel("Values", fontsize=label_fontsize)
            plt.title(wrapped_title, fontsize=label_fontsize)
            plt.legend()
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.grid(True)

            if save_plot:
                if image_path_png:
                    plt.savefig(
                        f"{image_path_png}/{save_filename}_time_series.png",
                        bbox_inches=bbox_inches,
                    )
                if image_path_svg:
                    plt.savefig(
                        f"{image_path_svg}/{save_filename}_time_series.svg",
                        bbox_inches=bbox_inches,
                    )

            plt.show()

        if plot_type in ["correlation", "both"]:
            plt.figure(figsize=(8, 6))
            sns.regplot(
                x=combined_df["Hashrate"],
                y=combined_df["Active Addresses"],
                ci=None,
                scatter_kws={"s": 10},
                line_kws={"color": "red"},
            )

            # Calculate equation of the line
            slope, intercept = np.polyfit(
                combined_df["Hashrate"], combined_df["Active Addresses"], 1
            )
            equation = (
                f"y = {slope:.2f}x "
                f"{'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
            )

            plt.xlabel("Bitcoin Hashrate", fontsize=label_fontsize)
            plt.ylabel("Litecoin Active Addresses", fontsize=label_fontsize)
            plt.title(wrapped_corr_title, fontsize=label_fontsize)

            # Plot an invisible line just for the legend
            plt.plot([], [], color="red", label=equation, linestyle="-")

            plt.legend(loc="upper left")
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.grid(True)

            if save_plot:
                if image_path_png:
                    plt.savefig(
                        f"{image_path_png}/{save_filename}_correlation.png",
                        bbox_inches=bbox_inches,
                    )
                if image_path_svg:
                    plt.savefig(
                        f"{image_path_svg}/{save_filename}_correlation.svg",
                        bbox_inches=bbox_inches,
                    )

            plt.show()

        return combined_df, correlation


################################################################################
