import pandas as pd


class Sum:
    def __init__(self, path):
        self.path = path

    def data_merger(self):
        ## 1_Remove unnecessary columns from 'electricity_con.csv' for further analysis

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

        # Save the filtered and modified DataFrame to a new CSV file
        # filtered_df.to_csv('filtered_electricity_con.csv', index=False)

        ## 2_Remove unnecessary columns from 'electricity_gen.csv', merge the previous file into 'merged_electricity_data.csv'and rename the columns

        # Read the original CSV file into a pandas DataFrame
        df = pd.read_csv(self.path[1])

        # Remove specified columns
        columns_to_drop = ["MSN", "Description", "Unit"]
        df.drop(columns=columns_to_drop, inplace=True)

        # Filter data based on conditions (YYYYMM from 200301 to 202306 and Column_Order is "1,2,3,11,12")
        filtered_df_gen = df[
            (df["YYYYMM"] >= 200301)
            & (df["YYYYMM"] <= 202306)
            & (df["Column_Order"].isin([1, 2, 3, 11, 12]))
        ]

        # Remove rows where MM is 13 in the 'YYYYMM' column
        filtered_df_gen = filtered_df_gen[filtered_df_gen["YYYYMM"] % 100 != 13]

        # Save the filtered DataFrame to a new CSV file
        # filtered_df_gen.to_csv('filtered_electricity_gen.csv', index=False)

        # Concatenate the two DataFrames
        merged_df = pd.concat([filtered_df_con, filtered_df_gen], ignore_index=True)

        # Save the merged DataFrame to a new CSV file
        # merged_df.to_csv('merged_electricity_data.csv', index=False)

        ## 3_Modified 'merged_electricity_data.csv

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

        # Save the modified DataFrame back to the same CSV file, overwriting the original data
        # pivoted_df.to_csv('merged_electricity_data.csv', index=False)

        ## 4_Dropped a Specific Row due to a invalid data:

        # Drop the row where the 'YYYYMM' column has the value '202203'
        df = pivoted_df[pivoted_df["YYYYMM"] != 202203]

        # Save the modified DataFrame back to the CSV file
        df.to_csv("merged_electricity_data.csv", index=False)
        return df
