import spacy
import pandas as pd

# Load the SciSpaCy models
nlp_sci_sm = spacy.load("en_core_sci_sm")
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")

# Read the combined text file
with open('combined_texts.txt', 'r') as file:
    text = file.read()

# Function to extract entities using SpaCy models
def extract_entities(nlp_model, text):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Extract entities using both models
entities_sci_sm = extract_entities(nlp_sci_sm, text)
entities_bc5cdr = extract_entities(nlp_bc5cdr, text)

# Convert to DataFrame for comparison
df_sci_sm = pd.DataFrame(entities_sci_sm, columns=['Entity', 'Label'])
df_bc5cdr = pd.DataFrame(entities_bc5cdr, columns=['Entity', 'Label'])

# Compare the entities from the two models
comparison = pd.concat([df_sci_sm, df_bc5cdr], keys=['SciSpaCy', 'BC5CDR'])
comparison.to_csv('entity_comparison.csv')

# Save the comparison to a CSV file
comparison.to_csv('entity_comparison.csv', index=False)

print("Entity comparison saved to 'entity_comparison.csv'")
