from .part_1_summary import Sum
from .part_4_Inference import EnergyAnalysis
def main():
    print("hello Professor Stefanie")
    path = [
    "https://raw.githubusercontent.com/lowei1023/Sustainable_Energy_Transition_Analysis/main/electricity_con.csv",
    "https://raw.githubusercontent.com/lowei1023/Sustainable_Energy_Transition_Analysis/main/electricity_gen.csv"
    ]
    merger = Sum(path)
    df = merger.data_merger()
    Inference = EnergyAnalysis(df)
    Inference.task_1()



main()
