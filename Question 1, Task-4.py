import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import spacy
import csv
from collections import Counter
from tqdm import tqdm

# Load the scispaCy models for biomedical NER
ner_model_sci = spacy.load('en_core_sci_sm')
ner_model_bc5cdr = spacy.load('en_ner_bc5cdr_md')

# Increase the max_length limit for both models if necessary
ner_model_sci.max_length = 1500000  # Adjust as needed
ner_model_bc5cdr.max_length = 1500000  # Adjust as needed

# Path to the input text file
input_text_file_path = r'combined_texts.txt'

# Read the text from the cleaned text file
with open(input_text_file_path, 'r', encoding='utf-8') as text_file:
    biomedical_text = text_file.read()

# Split the text into chunks (consider smaller sizes for better performance)
chunk_size = 1500000
text_chunks = [biomedical_text[i:i + chunk_size] for i in range(0, len(biomedical_text), chunk_size)]

# Print the total number of chunks
total_chunks = len(text_chunks)
print(f'Total number of chunks: {total_chunks}')

# Initialize counters for diseases and drugs for both models
diseases_counts_sci = Counter()
drugs_counts_sci = Counter()
diseases_counts_bc5cdr = Counter()
drugs_counts_bc5cdr = Counter()

# Function to process chunks and extract entities
def extract_entities(model, counter_diseases, counter_drugs, chunks):
    for chunk in tqdm(chunks, desc="Processing Chunks", unit="chunk"):
        doc = model(chunk)

        # Extract tokens and their entity types from the biomedical NER model output
        tokens_entities = [(token.text, token.ent_type_) for token in doc.ents]

        # Separate diseases and drugs
        diseases = [token[0] for token in tokens_entities if token[1] == 'DISEASE' or token[1] == 'DISEASES']
        drugs = [token[0] for token in tokens_entities if token[1] == 'DRUG' or token[1] == 'DRUGS']

        # Update counters
        counter_diseases.update(diseases)
        counter_drugs.update(drugs)

# Process chunks with both models
extract_entities(ner_model_sci, diseases_counts_sci, drugs_counts_sci, text_chunks)
extract_entities(ner_model_bc5cdr, diseases_counts_bc5cdr, drugs_counts_bc5cdr, text_chunks)

# Order entries by count in descending order
ordered_diseases_sci = [(word, count) for word, count in diseases_counts_sci.most_common()]
ordered_drugs_sci = [(word, count) for word, count in drugs_counts_sci.most_common()]
ordered_diseases_bc5cdr = [(word, count) for word, count in diseases_counts_bc5cdr.most_common()]
ordered_drugs_bc5cdr = [(word, count) for word, count in drugs_counts_bc5cdr.most_common()]

# Save results to CSV files
output_csv_file_path_sci = r'NEResult_sci.csv'
output_csv_file_path_bc5cdr = r'NEResult_bc5cdr.csv'

def save_results_to_csv(output_path, ordered_diseases, ordered_drugs):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Entity Type', 'Word', 'Count'])

        # Write diseases entries to CSV
        for word, count in ordered_diseases:
            csv_writer.writerow(['Disease', word, count])

        # Write drugs entries to CSV
        for word, count in ordered_drugs:
            csv_writer.writerow(['Drug', word, count])

save_results_to_csv(output_csv_file_path_sci, ordered_diseases_sci, ordered_drugs_sci)
save_results_to_csv(output_csv_file_path_bc5cdr, ordered_diseases_bc5cdr, ordered_drugs_bc5cdr)

print(f'Ordered word counts saved to {output_csv_file_path_sci} and {output_csv_file_path_bc5cdr}')

# Compare the results
def compare_results(counts_sci, counts_bc5cdr):
    total_diseases_sci = sum(counts_sci.values())
    total_diseases_bc5cdr = sum(counts_bc5cdr.values())
    
    print(f'Total diseases detected by SciSpacy: {total_diseases_sci}')
    print(f'Total diseases detected by BC5CDR: {total_diseases_bc5cdr}')
    print(f'Difference in total diseases: {total_diseases_bc5cdr - total_diseases_sci}')

    # Most common entities
    common_diseases_sci = counts_sci.most_common(5)
    common_diseases_bc5cdr = counts_bc5cdr.most_common(5)
    
    print(f'Most common diseases in SciSpacy: {common_diseases_sci}')
    print(f'Most common diseases in BC5CDR: {common_diseases_bc5cdr}')

compare_results(diseases_counts_sci, diseases_counts_bc5cdr)
