# CS 4650 Moral Stories
This repository provides code for preprocessing and classifying actions using the Moral Stories Dataset.

Structure:
* src/
    * dataset.py: Contains code for loading & tokenizing dataset
    * eval_utils.py: Contains code for calculating F-1 score and Accuracy
    * models.py: Contains code for our final LSTM
    * preprocess.py: Contains code for cleaning text 
* morality.ipynb: Main notebook for training & evaluating models

## Download Instructions
To get the dataset, download the zip file available at: https://github.com/demelin/moral_stories

Next, get the minimal pair split from action+context+consequence with the files: train.jsonl, test.jsonl, and dev.jsonl and place these files in the root directory.

## Replicate Results
Results reported in our paper can be replicated by running the morality.ipynb notebook located in the root folder. The final test accuracy and f1 scores are reported under the "Test Metrics" section.

We also provide code for obtaining results on baseline models (Naive Bayes, Logistic Regression, etc.) by running the "Baseline Models" section of the notebook.
