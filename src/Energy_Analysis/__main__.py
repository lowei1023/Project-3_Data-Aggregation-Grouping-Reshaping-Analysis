from .part_4_inference import EnergyAnalysis
import pandas as pd

def main():
    print("hello Professor Stefanie")
    path = "https://raw.githubusercontent.com/lowei1023/Sustainable_Energy_Transition_Analysis/main/merged_electricity_data.csv"
    df = pd.read_csv(path)
    Inference = EnergyAnalysis(df)
    Inference.task_1()
    Inference.task_2()
    Inference.task_5()
    Inference.task_6()



main()
