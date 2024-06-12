import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import ttest_ind
from itertools import combinations

def create_input_file(protein_file, meta_file) -> None:
    """
    Creates an input file by merging protein and metadata.
    This function reads protein and metadata files, merges them, and creates a final input file.
    Parameters:
        protein_file (str): Path to the protein data file.
        meta_file (str): Path to the metadata file.
    Returns:
        None
    Raises:
        FileNotFoundError: If either protein_file or meta_file is not found.
    """

    directory = os.path.dirname(meta_file)
    interim_file = os.path.join(directory, 'interim_data.csv')
    final_file = os.path.join(directory, 'final_data.csv')

    def add_genes_to_metadata(protein_file, meta_file, interim_file):
        # Read protein and meta data
        protein_data = pd.read_csv(protein_file, sep='\t')
        meta_data = pd.read_csv(meta_file)

        # Extract 'Genes' column from protein data and transpose
        genes = protein_data['Genes']
        genes_df = pd.DataFrame(genes).T

        # Set the 'Genes' as column names
        genes_df.columns = genes_df.iloc[0]
        
        # Drop the first row
        genes_df = genes_df.iloc[1:]

        # Concatenate meta data with genes data and write result to an interim CSV file
        merged_data = pd.concat([meta_data, genes_df], axis=1)
        merged_data.to_csv(interim_file, index=False)

    def add_protein_concentrations(meta_file, protein_file, final_file):
    # Read interim and protein data
        interim_data = pd.read_csv(meta_file)
        protein_data = pd.read_csv(protein_file, sep='\t')

        # Create an empty dataframe to store final data
        final_data = interim_data.copy()

        # Get unique sample names from the 'Sample' column
        sample_names = interim_data['SampleIDalt2'].unique().tolist()
            
        for sample_name in sample_names:
            # Find the row index where the sample name matches in interim data
            sample_row_index = final_data.index[final_data['SampleIDalt2'] == sample_name].tolist()

            # Check if the sample name exists in any column of protein data
            for column_name in protein_data.columns:
                column_name = str(column_name)
                if str(sample_name) in column_name:
                    # Extract protein concentrations for the sample
                    protein_concentrations = protein_data[column_name]
                    # Transpose and insert the concentrations into the final data
                    if sample_row_index:  # Check if sample_row_index is not empty
                        # Transpose the protein_concentrations without the header
                        transposed_column = pd.DataFrame(protein_concentrations).T

                        # Insert the transposed column into the final data starting from the 16th column
                        final_data.iloc[sample_row_index[0], 15:] = transposed_column

        # Write the final data to a CSV file
        final_data.to_csv(final_file, index=False)

    add_genes_to_metadata(protein_file, meta_file, interim_file)
    add_protein_concentrations(interim_file, protein_file, final_file)

    os.remove(interim_file)


def read_csv_get_pd(path: str) -> pd.DataFrame:
    ''' 
    Reads a CSV file and returns a pandas DataFrame. 
    Parameters:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file with blank cells filled with NaN.
    '''

    # Read data
    csv_data = path
    pd_csv_data = pd.read_csv(csv_data)         # Convert data into a Pandas DataFrame
    pd_csv_data = pd_csv_data.fillna(pd.NA)      # Fill empty cells with NaN

    return pd_csv_data


def generate_loess_plots(dataframe, path) -> None:
    '''
    Plots and saves LOESS functions from a given pandas dataframe.
    Parameters:
        dataframe (pd.DataFrame): Pandas DataFrame containing protein intensity data.
        path (str): Path to the CSV file from which the dataframe was generated.
    Returns:
        None
    Raises:
        None
    This function plots LOESS curves with shaded regions for standard deviation and averages concentrations for the first and second half of the timeframe.
    Plots are saved in the 'results_LOESS' folder within the specified path.
    A text file named 'excluded_proteins_LOESS.txt' is created in the 'results_LOESS' folder, listing proteins excluded due to less than 10% available data.
    '''

    # Specify the path to save the results folder and excluded proteins text file
    result_path = os.path.join(os.path.dirname(path), "results_LOESS")
    excluded_proteins_file = os.path.join(result_path, "excluded_proteins_LOESS.txt")

    os.makedirs(result_path, exist_ok=True)  # Create 'result' folder if it doesn't exist

    # exclude proteins with less then 10% available data
    excluded_proteins = []
    total_possible_data = len(dataframe['SampleIDalt2'].unique())
    for column in dataframe.columns[15:]:  # Start from the proteins column
        if len(dataframe[column].dropna()) < 0.1 * total_possible_data:  # Check if less than 10% data available
            excluded_proteins.append(column)  # Add protein to excluded list
            continue  # Skip if less than 10% data available

        # Group by 'HoL' and calculate mean LFQ and standard deviation for each hour for each protein
        average_concentration_per_hour = dataframe.groupby('HOL')[column].mean()
        std_deviation_per_hour = dataframe.groupby('HOL')[column].std()  # Calculate standard deviation

        # Calculate LOESS regression
        lowess = sm.nonparametric.lowess
        smoothed_data = lowess(average_concentration_per_hour.values, average_concentration_per_hour.index, frac=0.1)

        # Plot data, LOESS regression, and shaded region for standard deviation
        # plt.boxplot(dataframe.groupby('HOL')[column].apply(list), positions=average_concentration_per_hour.index, widths=0.5)
        plt.plot(smoothed_data[:, 0], smoothed_data[:, 1], label='LOESS Regression', color='red')
        plt.fill_between(average_concentration_per_hour.index, average_concentration_per_hour.values - std_deviation_per_hour.values,
                         average_concentration_per_hour.values + std_deviation_per_hour.values, alpha=0.3, label='Standard Deviation')

        # Calculate and plot points for DoL min and DoL max
        dol_groups = sorted(dataframe['DOL'].unique())
        min_dol = dol_groups[0]
        max_dol = dol_groups[-1]

        hol_groups = sorted(dataframe['HOL'].unique())
        min_hol = hol_groups[0]
        max_hol = hol_groups[-1]

        for dol_value in [min_dol, max_dol]:
            dol_average = dataframe[dataframe['DOL'] == dol_value][column].mean()
            hoL_value = (min_hol * 1.08) if dol_value == min_dol else (max_hol * 0.98) # adjust average presenting dots for visual purposes
            plt.scatter(hoL_value, dol_average, color='green' if dol_value == min_dol else 'blue', label=f'DoL {dol_value} Average', marker='o')

        # Adjust y-axis limits to add padding
        y_lower_limit = dataframe[column].min() - 0.1 * dataframe[column].min()  # 10% below the minimum value
        y_upper_limit = dataframe[column].max() + 0.1 * dataframe[column].max()  # 10% above the maximum value
        plt.ylim(y_lower_limit, y_upper_limit)

        plt.xlabel('HoL')
        plt.ylabel(r'Mean $\log_2$(LFQ)')
        plt.title(f'{column}')
        plt.legend()
        plt.savefig(os.path.join(result_path, f"{column}_LOESS-plot.png"))  # Save the plot in the 'result' folder
        plt.close()  # Close the plot to avoid displaying it multiple times

    # Write excluded proteins to text file
    with open(excluded_proteins_file, 'w') as file:
        file.write("Excluded Proteins:\n")
        for protein in excluded_proteins:
            percent_data_available = (len(dataframe[protein].dropna()) / total_possible_data) * 100
            file.write(f"{protein}: Only {percent_data_available:.2f}% of total data available\n")


def generate_volcano_plot(data: pd.DataFrame, path: str) -> None:
    """
    Generates a volcano plot to visualize protein significance.
    Parameters:
        data (pd.DataFrame): DataFrame containing protein intensity data.
        path (str): Path to save the results folder.
    Returns:
        None
    Raises:
        None
    This function generates a volcano plot to visualize the significance of protein ontogeny based on their intensity data.
    The plot is saved in the 'results_volcano' folder within the specified path.
    A text file named 'excluded_proteins_volcano.txt' is created in the 'results_volcano' folder, listing proteins excluded due to less than 10% available data.
    Another text file named 'significant_proteins.txt' is created in the 'results_volcano' folder, listing significant proteins with their corresponding p-values (generated from Welch Test).
    """

    # Specify the path to save the results folder
    result_path = os.path.join(os.path.dirname(path), "results_volcano")
    os.makedirs(result_path, exist_ok=True)
    excluded_proteins_file = os.path.join(result_path, "excluded_proteins_volcano.txt")

    # Calculate total possible data
    total_possible_data = len(data['SampleIDalt2'].unique())

    # Initialize list to store excluded proteins
    excluded_proteins = []

    # Calculate difference of mean protein intensity from DoL0 and DoL1 per protein
    differences = {}
    p_values = {}
    significant_proteins = []
    dol_groups = data['DOL'].unique()

    for protein in data.columns[15:]:
        intensity_per_DoL_per_protein = {}
        for dol_group in dol_groups:
            dol_data = data[data['DOL'] == int(dol_group)][protein]
            dol_mean = dol_data.mean()
            intensity_per_DoL_per_protein[dol_group] = dol_mean

        difference = list(intensity_per_DoL_per_protein.values())[-1] - list(intensity_per_DoL_per_protein.values())[0]
        # Welch's t-test
        _, p_value = ttest_ind(data[data['DOL'] == int(dol_groups[0])][protein],
                            data[data['DOL'] == int(dol_groups[-1])][protein],
                            equal_var=False, nan_policy='omit')  # equal_var=False for Welch's t-test

        differences[protein] = difference
        p_values[protein] = p_value

        # Check if protein has less than 10% data available
        if len(data[protein].dropna()) < 0.1 * total_possible_data:
            excluded_proteins.append(protein)
        else:
            if p_value < 0.05:
                significant_proteins.append(protein)

        
    # Create volcano plot
    plt.figure(figsize=(20, 9))
    for protein, difference in differences.items():
        p_value = p_values[protein]
        if p_value < 0.05:  # Check significance level
            plt.scatter(difference, -np.log10(p_value), label=protein)  # Label significant proteins in plot
            significant_proteins.append(protein)  # Add significant protein to the list
        else:
            plt.scatter(difference, -np.log10(p_value), label=None)  # Don't label others

    plt.xlabel(f'Difference in Mean Protein Intensity ($\log_2$(DoL{dol_groups[-1]}) - $\log_2$(DoL{dol_groups[0]}))')
    plt.ylabel(r'$-\log_{10}$(p-value)')
    plt.title('Volcano Plot for Proteins')
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='Significance Threshold (p=0.05)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper right')  

    # Save plot
    plt.savefig(os.path.join(result_path, "_proteins_volcano_plot.png"))
    plt.close()  # Close the plot to avoid displaying it multiple times

    # Write excluded proteins to text file
    with open(excluded_proteins_file, 'w') as file:
        file.write("Excluded Proteins:\n")
        for protein in excluded_proteins:
            percent_data_available = (len(data[protein].dropna()) / total_possible_data) * 100
            file.write(f"{protein}: Only {percent_data_available:.2f}% of total data available\n")

    # Write significant proteins to text file with p-values
    significant_proteins_file = os.path.join(result_path, "significant_proteins.txt")
    with open(significant_proteins_file, 'w') as file:
        file.write("Significant Proteins:\n")
        for protein in significant_proteins:
            p_value = p_values[protein]
            file.write(f"{protein}: p-value = {p_value}\n")


def generate_loess_plots_2factors(dataframe, path, fstfactor, scndfactor) -> None:
    '''
    Plot and save LOESS functions with two factors from a given pandas dataframe.
    Parameters:
        dataframe (pd.DataFrame): Pandas DataFrame containing protein intensity data.
        path (str): Path to the CSV file from which the dataframe was generated.
        fstfactor (str): Name of the first factor for grouping.
        scndfactor (str): Name of the second factor for grouping.
    Returns:
        None
    Raises:
        None
    This function plots LOESS curves with shaded regions for standard deviation and averages concentrations grouped by two factors.
    Plots are saved in the 'results_LOESS_{fstfactor}_{scndfactor}factor' folder within the specified path.
    A text file named 'excluded_proteins_LOESS.txt' is created in the 'results_LOESS_{fstfactor}_{scndfactor}factor' folder, listing proteins excluded due to less than 10% available data.
    '''

    fstfactor = fstfactor
    scndfactor = scndfactor

    # Specify the path to save the results folder and excluded proteins text file
    result_path = os.path.join(os.path.dirname(path), f"results_LOESS_{fstfactor}_{scndfactor}factor")
    excluded_proteins_file = os.path.join(result_path, "excluded_proteins_LOESS.txt")

    os.makedirs(result_path, exist_ok=True)  # Create 'result' folder if it doesn't exist

    # exclude proteins with less then 10% available data
    excluded_proteins = []
    total_possible_data = len(dataframe['SampleIDalt2'].unique())
    for column in dataframe.columns[15:]:  # Start from the proteins column
        if len(dataframe[column].dropna()) < 0.1 * total_possible_data:  # Check if less than 10% data available
            excluded_proteins.append(column)  # Add protein to excluded list
            continue  # Skip if less than 10% data available

        # Group by 'HoL' and 'Sex' and calculate mean LFQ and standard deviation for each hour for each protein
        grouped_data = dataframe.groupby([fstfactor, scndfactor])[column]
        average_concentration = grouped_data.mean()
        std_deviation = grouped_data.std()  # Calculate standard deviation

        # Plot data, LOESS regression, and shaded region for standard deviation for each sex
        for snd_factor_group in average_concentration.unstack().columns:
            average_concentration_per_fst_factor = average_concentration.unstack()[snd_factor_group]
            std_deviation_per_fst_factor = std_deviation.unstack()[snd_factor_group]

            # Calculate LOESS regression
            lowess = sm.nonparametric.lowess
            smoothed_data = lowess(average_concentration_per_fst_factor.values, average_concentration_per_fst_factor.index, frac=0.1)

            # Plot data, LOESS regression, and shaded region for standard deviation
            plt.plot(smoothed_data[:, 0], smoothed_data[:, 1], label=f'LOESS Regression ({snd_factor_group})')
            plt.fill_between(average_concentration_per_fst_factor.index, average_concentration_per_fst_factor.values - std_deviation_per_fst_factor.values,
                             average_concentration_per_fst_factor.values + std_deviation_per_fst_factor.values, alpha=0.3, label=f'Standard Deviation ({snd_factor_group})')

        # Adjust y-axis limits to add padding
        y_lower_limit = dataframe[column].min() - 0.1 * dataframe[column].min()  # 10% below the minimum value
        y_upper_limit = dataframe[column].max() + 0.1 * dataframe[column].max()  # 10% above the maximum value
        plt.ylim(y_lower_limit, y_upper_limit)

        plt.xlabel(fstfactor)
        plt.ylabel(r'Mean $\log_2$(LFQ)')
        plt.title(f'{column}')
        plt.legend()
        plt.savefig(os.path.join(result_path, f"{column}_LOESS-2ndfactor-plot.png"))  # Save the plot in the 'result' folder
        plt.close()  # Close the plot to avoid displaying it multiple times

    # Write excluded proteins to text file
    with open(excluded_proteins_file, 'w') as file:
        file.write("Excluded Proteins:\n")
        for protein in excluded_proteins:
            percent_data_available = (len(dataframe[protein].dropna()) / total_possible_data) * 100
            file.write(f"{protein}: Only {percent_data_available:.2f}% of total data available\n")


def significance_loess_2factors(dataframe, path, fstfactor, scndfactor, alpha=0.05) -> None:
    '''
    Determinates significant differences in protein concentrations grouped by two factors and save results to a text file.
    Parameters:
        dataframe (pd.DataFrame): Pandas DataFrame containing protein intensity data.
        path (str): Path to the CSV file from which the dataframe was generated.
        fstfactor (str): Name of the first factor for grouping.
        scndfactor (str): Name of the second factor for grouping.
        alpha (float, optional): Significance level. Defaults to 0.05.
    Returns:
        None
    Raises:
        None
    This function determines significant differences in protein concentrations between groups defined by two factors using Welch's t-test.
    Results are saved to a text file named 'significant_LOESS.txt' in the directory containing the CSV file.
    '''
    
    # Specify the path to save the results.txt
    significance_txt = os.path.join(os.path.dirname(path), "significant_LOESS.txt") 

    # Write significant outcomes
    with open(significance_txt, 'w') as file:
        file.write(f"Significant Differences for each {fstfactor} and {scndfactor}:\n\n")

        total_possible_data = len(dataframe['SampleIDalt2'].unique())

        for column in dataframe.columns[15:]:  # Start from the proteins column
            if len(dataframe[column].dropna()) < 0.1 * total_possible_data:  # Check if less than 10% data available
                continue  # Skip if less than 10% data available

            # Iterate over unique values of fstfactor
            for fst_factor_value in dataframe[fstfactor].unique():
                # Get data for the current fst_factor_value
                fst_group_data = dataframe[dataframe[fstfactor] == fst_factor_value]

                # Get unique values of scndfactor
                scnd_values = fst_group_data[scndfactor].unique()

                # Compare all combinations of scndfactor values
                for group1, group2 in combinations(scnd_values, 2):
                    # Get data for group1 and group2
                    group1_data = fst_group_data[fst_group_data[scndfactor] == group1][column]
                    group2_data = fst_group_data[fst_group_data[scndfactor] == group2][column]

                    # Perform Welch's t-test (as not same length of data)
                    _, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

                    # Check for significance
                    if p_value < alpha:
                        file.write(f"{column}: \n For {fstfactor} {fst_factor_value}, protein concentration between {group1} and {group2} differs significantly: p-value = {p_value}\n")