#Import necessary modules

from realtabformer import REaLTabFormer
from realtabformer import rtf_validators as rtf_val
from transformers import logging as hf_logging
import pandas as pd
## Set Hugging Face Transformers logging verbosity to error level
hf_logging.set_verbosity_error()
print("enter your saved realtabmodel path :")
path=input()

rtf_model= REaLTabFormer.load_from_dir(
    path="/workspaces/Generative-AI_recreating_football_game/rtf_checkpoints")

# Define a function named 'generate' that takes 'match_length' as a parameter
def generate(match_length):
    # Convert 'match_length' from minutes to seconds
    match_length_in_secs = float(match_length * 60)

    # Generate an initial sample using the REaLTabFormer model
    samples_validated = rtf_model.sample(
        n_samples=1, gen_batch=1,
        validator=obs_validator,
    )

    # Continue generating samples until the total duration is at least 'match_length_in_secs'
    while samples_validated.drop('label', axis=1).sum(axis=1).sum(axis=0) < match_length_in_secs:
        # Generate additional samples and concatenate them to 'samples_validated'
        samples_validated = pd.concat([samples_validated, rtf_model.sample(n_samples=1, gen_batch=1, validator=obs_validator)], ignore_index=True)

    # Return the generated samples
    return samples_validated

# Define a function named 'rest_data' that takes 'samples_validated' as a parameter
def rest_data(samples_validated):
    # Create an empty DataFrame named 'generated_df'
    generated_df = pd.DataFrame()

    # Inverse transform the 'label' column to its original values using the LabelEncoder 'le'
    generated_df['label'] = le.inverse_transform(samples_validated['label'])

    # Drop the 'label' column from 'samples_validated' and store the restof the data  in 't'
    t = samples_validated.drop('label', axis=1)

    # Convert the remaining columns in 't' to a list and store it in 'generated_df' under 'norm'
    generated_df['norm'] = t.values.tolist()

    # Remove any values of 0 and 100.0 from the 'norm' column in 'generated_df'
    generated_df['norm'] = generated_df['norm'].apply(lambda lst: [x for x in lst if x != 0 and x != 100.0])

    # Convert 'generated_df' to a list of dictionaries (records)
    generated_data = generated_df.to_dict('records')

    # Return the generated data
    return generated_data



print("How many minutes should the match be: ")
n=int(input())
match_mins=generate(n)
# apply the 'rest_data' function to 'samples_validated'
generated_data = rest_data(match_mins)

# Display the results
print (generated_data)

import json
json_data = json.dumps(generated_data)

with open("sample.json", "w") as outfile:
    outfile.write(json_data)
