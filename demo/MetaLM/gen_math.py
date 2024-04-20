import sys
import random

output=""
for _ in range(200):
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    if(random.random() < 0.5):
        c = a + b
        operator = '+'
    else:
        c = a - b
        operator = '-'
    output += f'{a}{operator}{b}={c};'
print(output)
print(len(output))
