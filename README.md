# Generative-AI_recreating_football_game

On a terminal, start with installing the requirement:

    ! pip install requirements.txt

If you want to retrain the model, run the script **train.py** (the data files have to be in the same folder as the script):

    ! python train.py

The **train.py** script contains all the pre-processing and training steps, and it saves the model at the end.

If you want to use the trained model, run the **generate.py** script

    ! python generate.py

You can look at the **footbar_usecase.ipynb** file for the code and the aproach explained, and you can also find a sample of the genrated data in **sample.json**

