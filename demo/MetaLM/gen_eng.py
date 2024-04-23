import os
import sys
import random

alphabet = ['a', 'b', 'c', 'd', 'e',
    'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y',
    'z']

with open(sys.argv[1], 'r') as f_reader:
    data = f_reader.read().strip().split("\n")
    words = [random.choice(data) for i in range(40)]
    string = ""
    for _ in range(400):
        word = random.choice(words)
        if(random.random() < 0.15):
            i = random.randint(0, len(word)-1)
            word = word[:i] + random.choice(alphabet) + word[i+1:]
        string += word + ";"
    print(string)
