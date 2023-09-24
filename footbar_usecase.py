#!/usr/bin/env python
# coding: utf-8

# Import the json module to work with JSON data
import json

# Open the first JSON file for reading and load its contents into the variable 'match1'
with open('/content/drive/MyDrive/footbar_usecase/match_1.json') as json_file:
    match1 = json.load(json_file)

# Open the second JSON file for reading and load its contents into the variable 'match2'
with open('/content/drive/MyDrive/footbar_usecase/match_2.json') as json_file:
    match2 = json.load(json_file)


# In[94]:


# Import the pandas library and alias it as 'pd'
import pandas as pd

# Create a DataFrame from the data in 'match1' using pandas and store it in 'df'
df = pd.DataFrame.from_dict(match1)

# Display the first few rows of the DataFrame
df.head()


# # Data preparation and EDA


# Calculate the length of the 'norm' column in the DataFrame 'df' and create a new column 'Length' to store the results
df['Length'] = df['norm'].str.len()


# Pad the lists with zeroes to make them of uniform size
df['norm'] = df['norm'].apply(lambda x: x + [0] * (df['Length'].max() - len(x)))




# Expand the lists into separate columns
df = pd.concat([df, df['norm'].apply(pd.Series)], axis=1)

# Drop the original 'norm' column
df = df.drop('norm', axis=1)



# comparing time taken for each movement/action in seconds :
df['time'] = df.drop(['label',"Length"],axis=1).sum(axis=0)


# Import necessary modules from sklearn for preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Initialize a LabelEncoder object
le = preprocessing.LabelEncoder()

# Use the LabelEncoder to transform the 'label' column in the DataFrame 'df'
df['label'] = le.fit_transform(df['label'])

# Create a dictionary that maps original class names to their encoded values
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Print the mapping of class names to their encoded values
print(le_name_mapping)


for i in range(100,227):
    df.drop(i, axis=1, inplace=True)




# transform the column names type to string ( this is needed for the realtabformer)
df.columns = df.columns.map(str)

get_ipython().system('pip install realtabformer -q')

from realtabformer import REaLTabFormer
from realtabformer import rtf_validators as rtf_val
from transformers import logging as hf_logging

## Set Hugging Face Transformers logging verbosity to error level
hf_logging.set_verbosity_error()


# Initialize a REaLTabFormer model with specific configurations
rtf_model = REaLTabFormer(
    model_type="tabular",  # Specify the model type as "tabular"
    batch_size=6,  # Set batch size to 6
    epochs=5,  # Specify the number of training epochs
    gradient_accumulation_steps=2,  # Set gradient accumulation steps
    logging_steps=5  # Specify logging steps during training
)




hf_logging.set_verbosity_error()
# fit the model to our tabular data and train it
rtf_model.fit(df, num_bootstrap=5)



rtf_model.save("rtf_model/")


