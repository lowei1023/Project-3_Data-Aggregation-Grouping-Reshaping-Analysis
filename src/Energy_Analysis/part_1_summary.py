import pandas as pd

class Sum:
    def __init__(self, path):
        """
        Initializes an instance of the Sum class.

        Parameters:
        path (list of str): The file paths to the CSV files.

        """
        self.path = path

    def data_merger(self):
        """
        Merges and processes data from two CSV files based on specific conditions and saves the result.

        Returns:
        pandas.DataFrame: The modified and merged DataFrame containing filtered and processed data.

        """
        # Read the original CSV file into a pandas DataFrame
        df = pd.read_csv(self.path[0])

        # Filter data based on conditions (YYYYMM from 200301 to 202306, Column_Order is "7", and MM is not 13)
        filtered_df_con = df[
            (df["YYYYMM"] >= 200301)
            & (df["YYYYMM"] <= 202306)
            & (df["Column_Order"] == 7)
            & (df["YYYYMM"] % 100 != 13)
        ]

        # Remove specified columns
        columns_to_drop = ["MSN", "Description", "Unit"]
        filtered_df_con.drop(columns=columns_to_drop, inplace=True)

        # Read the original CSV file into a pandas DataFrame
        df = pd.read_csv(self.path[1])

        # Remove specified columns
        df.drop(columns=columns_to_drop, inplace=True)

        # Filter data based on conditions (YYYYMM from 200301 to 202306 and Column_Order is "1,2,3,11,12")
        filtered_df_gen = df[
            (df["YYYYMM"] >= 200301)
            & (df["YYYYMM"] <= 202306)
            & (df["Column_Order"].isin([1, 2, 3, 11, 12]))
        ]

        # Remove rows where MM is 13 in the 'YYYYMM' column
        filtered_df_gen = filtered_df_gen[filtered_df_gen["YYYYMM"] % 100 != 13]

        # Concatenate the two DataFrames
        merged_df = pd.concat([filtered_df_con, filtered_df_gen], ignore_index=True)

        # Pivot the DataFrame to split the Column_Order values into separate columns
        pivoted_df = merged_df.pivot(
            index="YYYYMM", columns="Column_Order", values="Value"
        ).reset_index()

        # Rename the columns
        column_mapping = {
            1: "1_coal",
            2: "2_petroleum",
            3: "3_Natural_Gas",
            11: "11_solar",
            12: "12_wind",
            7: "7_consumption",
        }

        pivoted_df.columns = ["YYYYMM"] + [
            column_mapping.get(col, col) for col in pivoted_df.columns[1:]
        ]
        
        # Drop the row where the 'YYYYMM' column has the value '202203'
        df = pivoted_df[pivoted_df["YYYYMM"] != 202203]

        # Save the modified DataFrame back to the CSV file
        df.to_csv("merged_electricity_data.csv", index=False)
        return df
