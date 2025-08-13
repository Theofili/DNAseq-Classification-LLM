

import Bio
import re
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import Entrez
import io
import os
from collections import defaultdict

## Function to fetch sequences from differend dbs and differente title characteristics

class GetData(object):
  
  def email(email):

    Entrez.email = email 
  
  def get_data(db, title, size, output):
    

    # Make a query with function imputs
    print(f'Pulling sequences from the {db} database')
    print(f'Containing the word {title} in the title')
    print(f'Max number of sequneces is {size}')
    print(f'Name of fasta file: {output}.txt')

    handle = Entrez.esearch(db = db, term='Homo sapiens[ORGN] AND '+title+'[TITL]', retmax=size)


    record = Entrez.read(handle)
    
    id_list = record['IdList']

    # Fetch fasta corresponding to the ids

    res = Entrez.efetch(db=db, id=id_list, rettype='fasta', retmode='text')

    # Load and save fasta into a txt
    os.chdir('C:/Users/theos/SubCel/SubCel/data')

    with open(''+output+'.txt', 'w') as f:
        f.write(res.read())

    # Turn fasta into sequences

    data = defaultdict(list)
    with open(''+output+'.txt') as fp:
        for record in SeqIO.parse(fp, 'fasta'):
            sequence = str(record.seq)
            data['sequence'].append(sequence)


    df = pd.DataFrame.from_dict(data)
    df['type'] = title
    
    return(df)
  