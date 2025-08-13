


from Bio import Entrez
import os
from Bio import SeqIO
from collections import defaultdict
import pandas as pd
import statistics

# Pull protein sequences from human genome

handle = Entrez.esearch(db = 'protein', term='Homo sapiens[ORGN]', retmax=9999)

record = Entrez.read(handle)

id_list = record['IdList']

# Fetch the fasta files of the proteins hit

res = Entrez.efetch(db='protein', id=id_list, rettype='fasta', retmode='text')

# Load and save fasta into a txt

os.chdir('C:/Users/theos/SubCel/SubCel/data')

with open('average_length.txt', 'w') as f:
    f.write(res.read())

# Turn fasta into sequences

data = defaultdict(list)
with open('average_length.txt') as fp:
    for record in SeqIO.parse(fp, 'fasta'):
        sequence = str(record.seq)
        data['sequence'].append(sequence)

# Turn the dict into a dataframe - csv

df = pd.DataFrame.from_dict(data)
df.to_csv('all_proteins.csv')


# Drop duplicated sequences

df.drop_duplicates(subset='sequence', inplace=True)


# Add the sequences length as a column 

for sequnece in df.iterrows():
    len = df['sequence'].str.len()
    df['length'] = len

df.to_csv('filtered_proteins.csv', index=False)


longest_seq = df['sequence'].str.len().max()
shortest_seq = df['sequence'].str.len().min()

print(f'The longest sequence in the dataframe is: {longest_seq}')
print(f'The shortest sequence in the dataframe is: {shortest_seq}')



# Get statistics of the sequence lengths
distribution = df['length'].describe()
print(f'The distribution of the sequence lengths is: {distribution}')


# Calculate the average length of the sequences
lower = df['length'].quantile(0.05)  
upper = df['length'].quantile(0.95)

print(f'The lower bound of the sequence lengths is: {lower}')
print(f'The upper bound of the sequence lengths is: {upper}')

