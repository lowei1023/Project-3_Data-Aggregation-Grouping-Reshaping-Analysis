import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EnergyAnalysis:
    def __init__(self, df):
        self.df = df
        self.df["YYYYMM"] = pd.to_datetime(self.df["YYYYMM"], format="%Y%m")
        self.df["Year"] = self.df["YYYYMM"].dt.year
        self.df["Month"] = self.df["YYYYMM"].dt.month

    """
    Parameters:
    `df`: This is a pandas DataFrame containing the electricity consumption data. It is the primary input parameter for the `EnergyAnalysis` class constructor.

    Returns:
    None: The constructor (`__init__` method) initializes the class object and preprocesses the input DataFrame (`df`). It does not return any value. The processed DataFrame is stored as an instance variable (`self.df`) and can be accessed by other methods within the class.
    """

    def calculate_proportions(self):
        # Convert columns to numeric, replacing non-numeric values with NaN
        self.df["1_coal"] = pd.to_numeric(self.df["1_coal"], errors="coerce")
        self.df["2_petroleum"] = pd.to_numeric(self.df["2_petroleum"], errors="coerce")
        self.df["3_Natural_Gas"] = pd.to_numeric(
            self.df["3_Natural_Gas"], errors="coerce"
        )
        self.df["11_solar"] = pd.to_numeric(self.df["11_solar"], errors="coerce")
        self.df["12_wind"] = pd.to_numeric(self.df["12_wind"], errors="coerce")
        self.df["7_consumption"] = pd.to_numeric(
            self.df["7_consumption"], errors="coerce"
        )

        # Calculate proportions of renewable and traditional energy sources
        total_energy = self.df["7_consumption"]
        renewable_energy = self.df["11_solar"] + self.df["12_wind"]
        traditional_energy = (
            self.df["1_coal"] + self.df["2_petroleum"] + self.df["3_Natural_Gas"]
        )

        renewable_proportion = (renewable_energy / total_energy) * 100
        traditional_proportion = (traditional_energy / total_energy) * 100

        return renewable_proportion, traditional_proportion

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    `renewable_proportion`: This is a pandas Series representing the proportions of renewable energy (solar and wind) consumption relative to the total energy consumption. It is calculated as a percentage and indicates the proportion of energy derived from renewable sources.
    `traditional_proportion`: This is a pandas Series representing the proportions of traditional energy (coal, petroleum, and natural gas) consumption relative to the total energy consumption. It is calculated as a percentage and indicates the proportion of energy derived from traditional sources.

    Both `renewable_proportion` and `traditional_proportion` are returned by the method and can be used for further analysis or visualization.
    """

    def task_1(self):
        # Seaborn
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Year", y="7_consumption", data=self.df)
        plt.xlabel("Year")
        plt.ylabel("Electricity Consumption (GWh)")
        plt.title("Electricity Consumption Over Years (Seaborn)")
        plt.xticks(rotation=45)
        plt.gca().set_xticks(
            self.df["Year"].astype(int)
        )  # Set x-axis ticks as integer values
        plt.tight_layout()
        plt.show()

        # Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Year"].astype(int), self.df["7_consumption"])
        plt.xlabel("Year")
        plt.ylabel("Electricity Consumption (GWh)")
        plt.title("Electricity Consumption Over Years (Matplotlib)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates a line plot showing electricity consumption over the years and saves the plot as an image file named 'consumption_over_years.png' in the current working directory. The plot is displayed using `plt.show()` after saving the image.
    """

    def task_2(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Month",
            y="7_consumption",
            data=self.df.groupby("Month")["7_consumption"].mean().reset_index(),
        )
        plt.xlabel("Month")
        plt.ylabel("Average Electricity Consumption (GWh)")
        plt.title("Average Electricity Consumption per Month")
        plt.xticks(
            range(12),
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates a bar plot showing the average electricity consumption per month and saves the plot as an image file named 'average_consumption_per_month.png' in the current working directory. The plot is displayed using `plt.show()` after saving the image.
    """

    def task_3(self):
        # Seaborn
        energy_sources = [
            "1_coal",
            "2_petroleum",
            "3_Natural_Gas",
            "11_solar",
            "12_wind",
        ]
        plt.figure(figsize=(12, 6))
        for source in energy_sources:
            sns.lineplot(x=self.df["Year"], y=source, data=self.df, label=source)
        plt.xlabel("Year")
        plt.ylabel("Electricity Generation (GWh)")
        plt.title("Comparison of Different Energy Sources (Seaborn)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.gca().set_xticks(
            self.df["Year"].astype(int)
        )  # Set x-axis ticks as integer values
        plt.tight_layout()
        plt.show()

        # Matplotlib
        plt.figure(figsize=(12, 6))
        for source in energy_sources:
            plt.plot(self.df["Year"].astype(int), self.df[source], label=source)
        plt.xlabel("Year")
        plt.ylabel("Electricity Generation (GWh)")
        plt.title("Comparison of Different Energy Sources (Matplotlib)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates a line plot comparing the electricity generation from different energy sources over the years. The plot includes data for energy sources '1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', and '12_wind'. The plot is saved as an image file named 'energy_sources_comparison.png' in the current working directory. The plot is also displayed using `plt.show()` after saving the image.
    """

    def task_4(self):
        energy_sources = [
            "1_coal",
            "2_petroleum",
            "3_Natural_Gas",
            "11_solar",
            "12_wind",
        ]
        plt.figure(figsize=(14, 10))
        for i, source in enumerate(energy_sources, 1):
            plt.subplot(3, 2, i)
            sns.lineplot(x="Year", y=source, data=self.df)
            plt.xlabel("Year")
            plt.ylabel("Electricity Generation (GWh)")
            plt.title(f"{source} Generation Over Years")
        plt.tight_layout()
        plt.show()

        # Matplotlib
        plt.figure(figsize=(14, 10))
        for i, source in enumerate(energy_sources, 1):
            plt.subplot(3, 2, i)
            plt.plot(self.df["Year"].astype(int), self.df[source])
            plt.xlabel("Year")
            plt.ylabel("Electricity Generation (GWh)")
            plt.title(f"{source} Generation Over Years (Matplotlib)")
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates individual line plots for electricity generation from different energy sources ('1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind') over the years. Each plot represents the generation pattern for a specific energy source. The subplots are organized in a 3x2 grid. The entire plot is saved as an image file named 'individual_energy_sources.png' in the current working directory. The plot is also displayed using `plt.show()` after saving the image.
    """

    def task_5(self):
        sns.set(style="ticks")
        sns.pairplot(
            self.df[
                [
                    "7_consumption",
                    "1_coal",
                    "2_petroleum",
                    "3_Natural_Gas",
                    "11_solar",
                    "12_wind",
                ]
            ]
        )
        plt.suptitle("Pairplot for Correlation Analysis")
        plt.subplots_adjust(top=0.95)
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates a pair plot (scatterplot matrix) for the columns '7_consumption', '1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', and '12_wind' from the DataFrame. The pair plot is used for correlation analysis, showing the relationships between these variables. The plot is saved as an image file named 'pairplot.png' in the current working directory. The plot is also displayed using `plt.show()` after saving the image.
    """

    def task_6(self):
        # Seaborn
        total_energy = self.df[
            ["1_coal", "2_petroleum", "3_Natural_Gas", "11_solar", "12_wind"]
        ].sum(axis=1)
        self.df["Total_Energy"] = total_energy
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.df["Year"], y="Total_Energy", data=self.df)
        plt.xlabel("Year")
        plt.ylabel("Total Electricity Generation (GWh)")
        plt.title("Total Electricity Generation Over Years (Seaborn)")
        plt.xticks(rotation=45)
        plt.gca().set_xticks(
            self.df["Year"].astype(int)
        )  # Set x-axis ticks as integer values
        plt.tight_layout()
        plt.show()

        # Matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Year"].astype(int), self.df["Total_Energy"])
        plt.xlabel("Year")
        plt.ylabel("Total Electricity Generation (GWh)")
        plt.title("Total Electricity Generation Over Years (Matplotlib)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.
    Returns:
    None: This method does not return any values. It calculates the total electricity generation by summing the columns '1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', and '12_wind' along the rows and creates a new column 'Total_Energy' in the DataFrame to store these total values. Then, it generates a line plot showing the total electricity generation over the years. The plot is saved as an image file named 'total_energy_over_years.png' in the current working directory. The plot is also displayed using `plt.show()` after saving the image.
    """

    def task_7(self):
        correlation_matrix = self.df[
            [
                "7_consumption",
                "1_coal",
                "2_petroleum",
                "3_Natural_Gas",
                "11_solar",
                "12_wind",
            ]
        ].corr()
        consumption_correlation = correlation_matrix["7_consumption"][1:]
        print("Correlation between Consumption and Energy Sources:")
        print(consumption_correlation)

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:    
    None: This method does not return any values. It calculates the correlation matrix between the columns '7_consumption', '1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', and '12_wind'. It then extracts and prints the correlation values between '7_consumption' and the other energy sources ('1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind'). The method only prints the correlation values and does not return them or store them for further use.
    """

    def task_8(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates a heatmap of the correlation matrix for all columns in the DataFrame (including '7_consumption', '1_coal', '2_petroleum', '3_Natural_Gas', '11_solar', '12_wind'). The heatmap is saved as a file ('correlation_heatmap.png') and displayed to visualize the correlations between different columns in the dataset. The method only displays the heatmap and does not return or store it for further use.
    """

    def task_9(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Month", y="7_consumption", data=self.df)
        plt.xlabel("Month")
        plt.ylabel("Electricity Consumption (GWh)")
        plt.title("Seasonality in Electricity Consumption (Box Plots)")
        plt.xticks(
            range(12),
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )
        plt.tight_layout()
        plt.show()

    """
    Parameters:
    None: This method does not take any additional parameters as inputs.

    Returns:
    None: This method does not return any values. It generates box plots to display the seasonality in electricity consumption for each month. The box plots compare the distribution of '7_consumption' (electricity consumption in GWh) across different months. The plots are saved as a file ('seasonality_boxplots.png') and displayed to visualize the seasonal patterns in electricity consumption. The method only displays the box plots and does not return or store them for further use.
    """
