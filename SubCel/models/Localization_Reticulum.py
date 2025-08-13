

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



# Load the trained model and tokenizer

model_reticulum= AutoModelForSequenceClassification.from_pretrained("C:/Users/theos/SubCel/SubCel/models/model_reticulum")
tokenizer_reticulum = AutoTokenizer.from_pretrained("C:/Users/theos/SubCel/SubCel/models/model_reticulum")

def classify_protein_sequences_reticulum(sequence):
    sequence = sequence.upper()
    print(sequence)

    if not isinstance(sequence, str) or not sequence:
        return "Invalid input: Please provide a non-empty DNA sequence."

    if len(sequence) > tokenizer_reticulum.model_max_length:
        return f'Input sequence is too long. Maximum length is {tokenizer_reticulum.model_max_length}.'

    inputs = tokenizer_reticulum(sequence, return_tensors='pt', padding=True, truncation=True)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    device = model_reticulum.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_reticulum(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    class_labels = ['not membrane', 'membrane']  # adjust as needed

    if predicted_class_id < len(class_labels):
        return class_labels[predicted_class_id]
    else:
        return f'Unknown class ID: {predicted_class_id}'



# Example usage
print(classify_protein_sequences_reticulum('MGAGVGVAGCTRGHRNWVPSQLPPREIKAGVSLAVVTEFAWVLAPRPKRATASALGTESPRFLDRPDFFDYPDSDQARLLAVAQFIGEKPIVFINSGSSPGLFHHILVGLLVVAFFFLLFQFCTHINFQKGA'))