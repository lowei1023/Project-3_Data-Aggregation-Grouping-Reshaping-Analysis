import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class EDA:
    def __init__(self, df):
        
        """
    
        Initializes an instance of the EDA class.

        Parameters:
        df (pandas.DataFrame): The input DataFrame that will be used for analysis and visualization.
        
        """
        
        self.df = df
       
    
   
    
    def generate_summary_statistics(self):
        
        """
        Generate summary statistics for the input DataFrame.

        Returns:
        pandas.DataFrame: Summary statistics DataFrame generated using the describe() function applied to the input DataFrame.
        """
        
        return self.df.describe()
 


    def generate_histograms(self):
        """
        Generate and display histograms for each column in the DataFrame using Matplotlib and Seaborn libraries.

        Returns:
        None
        """
        
        # Generate histograms using Matplotlib
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)

   
        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            plt.hist(self.df[column], bins=20, color='skyblue', edgecolor='black')
            plt.title(column)
            plt.xlabel(column)
            plt.ylabel('Frequency')

        plt.show()
        
        # Generate histograms using Seaborn
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        
        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            sns.histplot(self.df[column], bins=20, kde=True, color='skyblue')
            plt.title(column)
            plt.xlabel(column)
            plt.ylabel('Density')

        plt.show()

    

    def generate_box_plots(self):
        """
        Generate and display boxplots for each column in the DataFrame.

        Returns:
        None
        """


        # Using Matplotlib:
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            plt.boxplot(self.df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
        plt.show()
    
        # Using Seaborn
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            sns.boxplot(data=self.df[column], color='skyblue')
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
        plt.show()




    def generate_scatter_plots(self):
        """
        Generate and display scatter plots for all combinations of columns in the DataFrame.

        Returns:
        None
        """


        # Using Seaborn:

  
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        sns.pairplot(self.df, kind='scatter')
        plt.show()


    def generate_correlation_heatmap(self):
        """
        Generate and display a correlation matrix heatmap for the DataFrame.

        Returns:
        None
        """
        # Using Matplotlib:

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title('Correlation Matrix Heatmap')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.show()


        # Using Seaborn:

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix Heatmap')
        plt.show()



   
    def generate_violin_plots(self):

        """
        Generate and display violin plots for each column in the DataFrame.

        Returns:
        None
        """

        # Using Matplotlib:


        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)
        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            plt.violinplot(self.df[column])
            plt.title(f'Violin plot of {column}')
            plt.xlabel(column)
        plt.show()


        # Using Seaborn:


        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5)

        for i, column in enumerate(self.df.columns[1:]):
            plt.subplot(3, 2, i + 1)
            sns.violinplot(data=self.df[column], color='skyblue')
            plt.title(f'Violin plot of {column}')
            plt.xlabel(column)

        plt.show()
