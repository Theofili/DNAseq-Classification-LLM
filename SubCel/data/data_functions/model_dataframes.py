import pandas as pd


filtered_df = pd.read_csv('filtered_protein_data.csv', index_col=0)

full_df = pd.read_csv('protein_data.csv', index_col=0)


def make_binary_type_df(full_df, target_type, sample_n=None, out_csv=None):
    
    # Filter for the target type and length

    df_target = filtered_df[filtered_df['type'] == target_type].copy()
    df_target = df_target.drop_duplicates('sequence')

    # Optionally sample from full_df for negatives

    if sample_n is not None:
        df_neg = full_df.sample(n=sample_n, random_state=42)
    else:
        df_neg = full_df

    # Concatenate and drop duplicates

    df_combined = pd.concat([df_target, df_neg])
    df_combined = df_combined.drop_duplicates('sequence')
    if 'length' in df_combined.columns:
        df_combined = df_combined.drop(columns=['length'])

    # Map types to binary

    df_combined['type'] = df_combined['type'].apply(lambda x: 1 if x == target_type else 0)

    # Optionally save to CSV

    if out_csv:
        df_combined.to_csv(out_csv, index=False)

    return df_combined




# Get 50:50 binary classification dataframes for each type

model_extracellular_df = make_binary_type_df(filtered_df, target_type='extracellular', sample_n =200, out_csv='model_extracellular_data.csv')
model_membrane_df = make_binary_type_df(filtered_df, target_type='transmembrane', sample_n=5000, out_csv='model_membrane_data.csv')
model_nuclear_df = make_binary_type_df(filtered_df, target_type='nuclear', sample_n=3000, out_csv='model_nuclear_data.csv')
model_mitochondrial_df = make_binary_type_df(filtered_df, target_type='mitochondrial', sample_n=3000, out_csv='model_mitochondrial_data.csv')
model_golgi_df = make_binary_type_df(filtered_df, target_type='golgi', sample_n=500, out_csv='model_golgi_data.csv')
model_reticulum_df = make_binary_type_df(filtered_df, target_type='reticulum', sample_n=500, out_csv='model_reticulum_data.csv')
model_ribosome_df = make_binary_type_df(filtered_df, target_type='ribosome', sample_n=200, out_csv='model_ribosome_data.csv')
model_lysosome_df = make_binary_type_df(filtered_df, target_type='lysosome', sample_n=5000, out_csv='model_lysosome_data.csv')
model_peroxisome_df = make_binary_type_df(filtered_df, target_type='peroxisome', sample_n=300, out_csv='model_peroxisome_data.csv')