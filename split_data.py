import random

with open('email_phishing_data2.csv', 'r') as f:
    lines = f.readlines()


def clean():
    res = []
    for line in lines:
        if line.split(',')[8].count('1') != 0:
            res.append(line)
        elif random.random() <= 0.015:
            res.append(line)
    with open('email_phishing_data.csv', 'w') as f:
        f.writelines(res)

def split():
    test = []
    train = []

    for line in lines:
        if random.random() <= 0.15:
            test.append(line)
        else:
            train.append(line)
    with open('test.csv', 'w') as f:
        f.writelines(test)
    with open('train.csv', 'w') as f:
        f.writelines(train)
    print(f'Test: {len(test)}')
    print(f'Train: {len(train)}')
    print(f'Split: {100.0 * len(test) / len(train)}%')
split()