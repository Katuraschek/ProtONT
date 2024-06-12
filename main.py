import utils
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# specify path to input file
path_to_csv_file = "C:/Users/CH258405/Documents/1-NeonatalBloodPlasmaProtein/input.csv"

# convert csv to pandas dataframe
pd = utils.read_csv_get_pd(path_to_csv_file)
pd['HOL'] = pd['HOL'].astype(int)

# log transform the intensity data
for column in pd.columns[15:]: 
    pd[column] = np.log2(pd[column])

# generate LOESS plots
utils.generate_loess_plots(pd, path_to_csv_file)

# normalization for intensity data
scaler = MinMaxScaler()
pd.iloc[:, 15:] = scaler.fit_transform(pd.iloc[:, 15:])

# generate Vulcano plots
utils.generate_volcano_plot(pd, path_to_csv_file)
