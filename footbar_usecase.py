#!/usr/bin/env python
# coding: utf-8

# After examining the two data files, I can see that the data can be considered as time series. Since the goal of this task is to generate similar time series or sequences, my first choice would be seq-to-seq models or transformers.
# 
# Informer and Autoformer from the Hugging Face library would be good solutions. They perform well for time series forecasting.
# 
# Another quick solution would be RealTabformer, a transformer model designed for synthetic tabular data generation. If I transform the JSON data into a pandas DataFrame and use the model to generate similar synthetic data, and then transform the data back to JSON, it should work.
# 
# Due to limited memory and time constraints, I will train the RealTabformer model using only one file since it requires less RAM and GPU resources, making it easier to train and use.

# # Importing data
# 

# In[49]:


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

# In[95]:


# Calculate the length of the 'norm' column in the DataFrame 'df' and create a new column 'Length' to store the results
df['Length'] = df['norm'].str.len()


# In[96]:


# Find the maximum length value in the 'Length' column
df['Length'].max()


# In[97]:


# Find the minimum length value in the 'Length' column
df['Length'].min()


# In[98]:


# Import the required libraries for the EDA part
import seaborn as sns  # Import seaborn for data visualization

# Set the style for seaborn plots
sns.set()

# Create a bar plot using seaborn to visualize and compare the length of list of time serie for each action
sns.barplot(data=df, x="Length", y="label")


# We can see here that the time series associated with the rest action are the longest. While the time series associated with the shot shot are the shortest.

# In[99]:


# Pad the lists with zeroes to make them of uniform size
df['norm'] = df['norm'].apply(lambda x: x + [0] * (df['Length'].max() - len(x)))
# check if it worked (Find the minimum length value in the 'Length' column)
df['Length'] = df['norm'].str.len()
df['Length'].min()


# In[100]:


# Expand the lists into separate columns
df = pd.concat([df, df['norm'].apply(pd.Series)], axis=1)

# Drop the original 'norm' column
df = df.drop('norm', axis=1)

# Print the resulting DataFrame
df


# In[ ]:


# comparing time taken for each movement/action in seconds :
df['time'] = df.drop(['label',"Length"],axis=1).sum(axis=0)


# In[102]:


sns.set()
sns.barplot(data=df, x="label", y="time")


# The "cross" action takes the least time while the "tackle" action takes the most.

# In[51]:


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


# In this part, I limited the players' movement length to 100 instead of 227 due to memory limitations. This should not be done if you have sufficient GPU and RAM. In optimal conditions I would use the second data file with movements of lengths = 700 .

# In[ ]:


for i in range(100,227):
    df.drop(i, axis=1, inplace=True)


# In[10]:


# transform the column names type to string ( this is needed for the realtabformer)
df.columns = df.columns.map(str)


# # Modeling

# 
# 
# <center>
#     <h1>REaLTabFormer</h1>
#     <br>
#     <div align="center" style="width:70%;">
#         <p style="font-size: 1.2em; text-align:justify">The REaLTabFormer (Realistic Relational and Tabular Data using Transformers) offers a unified framework for synthesizing tabular data of different types. A sequence-to-sequence (Seq2Seq) model is used for generating synthetic relational datasets. The REaLTabFormer model for a non-relational tabular data uses GPT-2, and can be used out-of-the-box to model any tabular data with independent observations.</p>
#     </div>
#     <br>
#     
# <div align="center" style="width:70%;">
#     <img src="https://github.com/avsolatorio/RealTabFormer/raw/main/img/REalTabFormer_Final_EQ.png" style="width:50%"/>
#  </div>  
#     <p align="center">
#     <strong>REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers</strong>
# <div align="center" style="width:70%;">           
#     <a href="https://arxiv.org/abs/2302.02041">Paper on ArXiv</a>
#     </p>
# 
# </div>
# </center>

# In[11]:


# install realtabformer package
get_ipython().system('pip install realtabformer -q')


# In[12]:


#Import necessary modules

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


# In[13]:


hf_logging.set_verbosity_error()
# fit the model to our tabular data and train it
rtf_model.fit(df, num_bootstrap=5)


# # Results

# In[14]:


# Create an instance of the ObservationValidator from realtabformer
obs_validator = rtf_val.ObservationValidator()

# Set Hugging Face Transformers logging verbosity to error level
hf_logging.set_verbosity_error()

# Generate samples using the REaLTabFormer model
# - n_samples: Number of samples to generate (200 in this case)
# - gen_batch: Batch size for generating samples (5 in this case)
# - validator: Use the ObservationValidator for validation
samples_validated = rtf_model.sample(
    n_samples=200,
    gen_batch=6,
    validator=obs_validator)


# In[15]:


# print the number of samples generated
samples_validated


# In[16]:


samples_validated['label'].unique()


# # Generate a fixed length sequence :

# In[34]:


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


# In[40]:


# generate a 60 mins long sample
match_mins_60=generate(60)


# In[43]:


# checking if the length is correct:
match_mins_60.drop('label',axis=1).sum(axis = 1).sum(axis = 0)//60


# In[46]:


# checking2 (20 mins):
match_mins_15=generate(20)
match_mins_15.drop('label',axis=1).sum(axis = 1).sum(axis = 0)//60


# # Restoring the original data format

# In[17]:


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


# In[18]:


# apply the 'rest_data' function to 'samples_validated'
generated_data = rest_data(samples_validated)

# Display the results
generated_data


# # Save model

# In[47]:


rtf_model.save("rtf_model/")

