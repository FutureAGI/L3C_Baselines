import sys
import random

output=""
for _ in range(400):
    a = random.randint(0, 5)
    b = random.randint(0, 4)
    if(random.random() < 0.5):
        c = a + b
        operator = '+'
        if(c > 9):
            continue
    else:
        c = a - b
        operator = '-'
        if(c < 0):
            continue
    if(random.random() < 0.15):
        c = random.choice(range(10))
    output += f'{a}{operator}{b}={c};'
print(output)
