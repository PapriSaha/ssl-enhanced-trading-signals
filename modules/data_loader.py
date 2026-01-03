import os  # Used for checking file existence and path manipulation
import pandas as pd  # Core library for data manipulation using DataFrames


class DataLoader:
    """
    Handles loading and initial column renaming of datasets.
    This class abstracts away the file reading logic to ensure consistency across training and testing.
    """

    def load_file(self, path):
        """
        Loads a CSV or Excel file based on extension.
        Different datasets might come in different formats (.csv vs .xlsx).
        """
        # Check if file exists to prevent crashing later with obscure errors
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # If it's Excel, use read_excel
        if path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(path)

        # Default to CSV reading
        return pd.read_csv(path)

    def rename_cols(self, df):
        """
        Standardizes column names to Open, High, Low, Close, Volume.
        Why: Raw data often has variations like 'Time', 'time', 'CLOSE', 'close'.
        The pipeline expects standard capitalization for feature generation.
        """
        # Map common variations to our standard names
        remap = {'time': 'Date', 'Time': 'Date', 'open': 'Open', 'high': 'High',
                 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}

        # Apply the renaming
        df.rename(columns=remap, inplace=True)

        # Return unique columns only (removes duplicate columns if any exist due to bad data)
        return df.loc[:, ~df.columns.duplicated()]

    def load_and_clean(self, path):
        """
        High-level function to load and clean a dataset.
        Why: The pipeline needs one simple call to get ready-to-use data.
        """
        print(f"Reading file: {path}")  # Log progress

        # 1. Load the raw file
        df = self.load_file(path)

        # 2. Fix column names immediately so we can access 'Date', 'Close', etc.
        df_clean = self.rename_cols(df)
        print(f"Data loaded: {df_clean.shape}")

        # 3. Standardize Date column if it exists
        # Why: Backtesting requires strict chronological order.
        if 'Date' in df_clean.columns:
            try:
                # Get first value to detect format
                first_val = df_clean['Date'].iloc[0]
                try:
                    # Check if it's a Unix timestamp (float/int)
                    float(first_val)
                    # Convert timestamp to datetime object
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], unit='s')
                except (ValueError, TypeError):
                    # If conversion fails, it's likely a string (e.g., "2023-01-01")
                    # 'coerce' turns unparseable data into NaT (Not a Time) rather than crashing
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            except Exception as e:
                # Log any date issues but don't stop execution if not critical
                print(f"  Note: Date parsing warning: {e}")

        return df_clean
