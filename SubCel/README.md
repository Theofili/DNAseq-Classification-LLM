

# Subcellular localization prediction based on a pre-trained Mistral model

## Collecting Data

In the data folder there are data_functions including:
* Pull data from the Entrez Database, based on user preference
* Create datasets for binary classification model training "one-vs-all"

In the data folders there are also programs related to computations for data normalization 


## Training the models

In the models folder there are finetuned models each with a different cellular compartment/place name which are trained to distinguish between sequence that belong in the place or not

The localization function is the one that the user can run to classify the sequence
