import pandas as pd
from sklearn.model_selection import train_test_split
import bz2
import pickle
import os
import torch
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import Tensor
from transformers import PreTrainedTokenizer, BatchEncoding
from typing import Literal, List, Dict, Tuple, Optional
from transformers import BatchEncoding
from autextication import Autextication
from corpus import Corpus

def compute_general_stats(data: List[Dict]):
    #print(src)
    samples = 0
    human_gen = 0
    ai_gen = 0
    for item in data:
        samples += 1
        if item['label'] == 0:
            human_gen += 1

        elif item['label'] == 1:
            ai_gen += 1

    ratio = human_gen/ai_gen

    # Compute average length of human-generated and AI-generated texts
    human_lengths = [len(item['text']) for item in data if item['label'] == 0]
    ai_lengths = [len(item['text']) for item in data if item['label'] == 1]
    human_avg_length = statistics.mean(human_lengths)
    ai_avg_length = statistics.mean(ai_lengths)

    # Compute standard deviation of length of human-generated and AI-generated texts
    human_std_dev = statistics.stdev(human_lengths)
    ai_std_dev = statistics.stdev(ai_lengths)

    print('\nthe number of samples is ' + str(samples))
    print('\nthe number of human generated samples is ' + str(human_gen))
    print('\nthe number of AI generated samples is ' + str(ai_gen))
    print('\nthe ratio between human and Ai generated samples is ' + str(ratio))

    # Plot histogram of human-generated text lengths
    plt.hist(human_lengths, bins=20)
    plt.title('Length of Human-Generated Texts')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.axvline(human_avg_length, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(human_avg_length + human_std_dev, color='red', linestyle='dotted', linewidth=1)
    plt.axvline(human_avg_length - human_std_dev, color='red', linestyle='dotted', linewidth=1)
    plt.show()

    plt.savefig('human_lengths.png')
    plt.close()

    # Plot histogram of AI-generated text lengths
    plt.hist(ai_lengths, bins=20)
    plt.title('Length of AI-Generated Texts')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.axvline(ai_avg_length, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(ai_avg_length + ai_std_dev, color='red', linestyle='dotted', linewidth=1)
    plt.axvline(ai_avg_length - ai_std_dev, color='red', linestyle='dotted', linewidth=1)
    plt.show()



    plt.savefig('ai_lengths.png')
    plt.close()


#def vocabulary(src: ):


