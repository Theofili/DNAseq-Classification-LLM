
from SubCel.data.data_functions.get_data import GetData
import pandas as pd

import matplotlib.pyplot as plt

# Download the data -- 6 subcellular locations

nuclear_df = GetData.get_data(db="protein",title='nuclear' , size=1000, output='nuclear')
transmembrane_df = GetData.get_data(db="protein", title='transmembrane', size=1000, output='membrane')
mitochondrial_df = GetData.get_data(db="protein", title='mitochondrial', size=1000, output='mitochondrial')
extracellular_df = GetData.get_data(db="protein", title='extracellular', size=1000, output='extracellular') 
golgi_df = GetData.get_data(db="protein", title='golgi', size=1000, output='golgi')
reticulum_df = GetData.get_data(db="protein", title='reticulum', size=1000, output='reticulum')


# Combine dataframes

all_proteins = pd.concat([nuclear_df, transmembrane_df, mitochondrial_df, extracellular_df, golgi_df, reticulum_df], ignore_index=True)


# Add the sequences length as a column

all_proteins['length'] = all_proteins['sequence'].str.len()

# Save the combined dataframe to a CSV file

all_proteins.to_csv('all_proteins_with_lengths.csv', index=False)

all_proteins.drop_duplicates('sequence')

# Make a plot of the average sequence length distribution

plt.hist(all_proteins['length'], bins=50, color='darkorange', alpha=0.7)
plt.title("Average Sequence Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.show()


# Make a plot of the average sequence length per subcellular location

plt.figure(figsize=(16, 6))

for type_ in all_proteins['type'].unique():
    type_df = all_proteins[all_proteins['type'] == type_]
    plt.hist(
        type_df['length'],
        bins=50,
        alpha=0.6,
        label=type_,
        histtype='stepfilled'
    )

plt.legend()
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("Sequence Length Distribution by Type")

# Highlight extracellular in a different color 
plt.hist(all_proteins[all_proteins['type'] == 'extracellular']['length'], bins=50, alpha=0.9, color='crimson', label='extracellular')

plt.show()