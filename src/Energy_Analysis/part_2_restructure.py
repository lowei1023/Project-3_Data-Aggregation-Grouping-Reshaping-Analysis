import pandas as pd


class DataRestructure:

    """
    A class for restructuring electricity consumption and generation data.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing electricity data.

    Attributes:
    df (pandas.DataFrame): The DataFrame containing electricity data.
    """

    def __init__(self, df):
        """
        Initializes the DataRestructure class with the provided DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing electricity data.
        """

        self.df = df
        self.df["YYYYMM"] = pd.to_datetime(self.df["YYYYMM"], format="%Y%m")
        self.df["Year"] = self.df["YYYYMM"].dt.year

    def calculate_yearly_data(self):
        """
        Calculate yearly aggregated data for different energy sources.

        Returns:
        pandas.DataFrame: Yearly aggregated data for energy sources (coal, petroleum, natural gas, solar, wind).
        """

        yearly_data = (
            self.df.groupby("Year")
            .agg(
                {
                    "1_coal": "sum",
                    "2_petroleum": "sum",
                    "3_Natural_Gas": "sum",
                    "7_consumption": "sum",
                    "11_solar": "sum",
                    "12_wind": "sum",
                }
            )
            .reset_index()
        )
        print(yearly_data)

    def calculate_average_generation(self):
        """
        Calculate the average generation of different energy sources per year.

        Returns:
        pandas.DataFrame: Average generation of energy sources per year.
        """

        average_generation = (
            self.df.groupby("Year")
            .agg(
                {
                    "1_coal": "mean",
                    "2_petroleum": "mean",
                    "3_Natural_Gas": "mean",
                    "11_solar": "mean",
                    "12_wind": "mean",
                }
            )
            .reset_index()
        )
        print(average_generation)

    def reshape_data(self):
        """
        Reshape the DataFrame by adding Date, Year, and Month columns and creating a pivot table.

        Returns:
        pandas.DataFrame: Pivot table with Year as index and Months as columns showing energy source values.
        """
        self.df["Date"] = pd.to_datetime(self.df["YYYYMM"], format="%Y%m")
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month

        pivot_data = self.df.pivot_table(
            index="Year",
            columns="Month",
            values=["1_coal", "2_petroleum", "3_Natural_Gas"],
        )
        print(pivot_data)

    def calculate_mean_sum_specific_columns(self):
        """
        Calculate the mean and sum values of specific energy sources (coal, petroleum, natural gas).

        Returns:
        pandas.Series: Mean values of specific energy sources.
        pandas.Series: Sum values of specific energy sources.
        """

        mean_values = self.df[["1_coal", "2_petroleum", "3_Natural_Gas"]].mean()
        sum_values = self.df[["1_coal", "2_petroleum", "3_Natural_Gas"]].sum()
        print("Mean values:")
        print(mean_values)
        print("\nSum values:")
        print(sum_values)

    def calculate_mean_by_year_sum_by_month(self):
        """
        Calculate mean values by year and sum values by month for specific energy sources.

        Returns:
        pandas.DataFrame: Mean values of specific energy sources by year.
        pandas.DataFrame: Sum values of specific energy sources by month.
        """

        mean_by_year = self.df.groupby("Year")[
            ["1_coal", "2_petroleum", "3_Natural_Gas"]
        ].mean()
        sum_by_month = self.df.groupby("Month")[
            ["1_coal", "2_petroleum", "3_Natural_Gas"]
        ].sum()
        print("Mean by Year:")
        print(mean_by_year)
        print("\nSum by Month:")
        print(sum_by_month)

    def reshape_stack_unstack(self):
        """
        Reshape the DataFrame by stacking and unstacking specific columns.

        Returns:
        pandas.Series: Stacked data with Year, Month, and specific energy sources.
        pandas.DataFrame: Unstacked data with Year as index and specific energy sources as columns.
        """

        stacked = self.df.set_index(["Year", "Month"])[
            ["1_coal", "2_petroleum"]
        ].stack()
        unstacked = stacked.unstack()
        print("Stacked Data:")
        print(stacked)
        print("\nUnstacked Data:")
        print(unstacked)
