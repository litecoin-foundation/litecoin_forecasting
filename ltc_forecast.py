if __name__ == "__main__":

    ## Import the necessary libraries
    import os
    import sys
    from main import CryptoCorrelation
    from eda_toolkit import ensure_directory

    # Add the directory containing the `main.py` file to the system path
    sys.path.append(os.pardir)

    base_path = os.path.join(os.pardir)

    # Go up one level from 'notebooks' to parent directory,
    # then into the 'data' folder
    data_path = os.path.join(os.pardir, "data")
    data_output = os.path.join(os.pardir, "data_output")

    # create image paths
    image_path_png = os.path.join(base_path, "images", "png_images")
    image_path_svg = os.path.join(base_path, "images", "svg_images")

    # Use the function to ensure'data' directory exists
    ensure_directory(data_path)
    ensure_directory(data_output)
    ensure_directory(image_path_png)
    ensure_directory(image_path_svg)

    ## Initialize
    crypto_correlation = CryptoCorrelation(api_key="your_coinmetrics_api_key")

    ## Fetch Litecoin active addresses
    ltc_active_addresses = crypto_correlation.fetch_coin_active_addresses(
        start_date="2023-01-01",
        end_date="2024-01-01",
        normalize=True,
    )

    ## Fetch Bitcoin hashrate
    btc_hashrate = crypto_correlation.fetch_bitcoin_hashrate(normalize=True)

    ## Analyze correlation and generate plots
    combined_df, correlation = crypto_correlation.connect_coin_users_to_btc_hashrate(
        start_date="2023-01-01",
        end_date="2024-01-01",
        normalize=True,
        plot_type="both",
        save_plot=True,
        image_path_png=image_path_png,
        save_filename="ltc_btc_correlation",
    )

    ## View the results
    print(combined_df)
    print(f"Correlation: {correlation:.4f}")
    ## Initialize the CryptoCorrelation class
    crypto_correlation = CryptoCorrelation(api_key="your_coinmetrics_api_key")
