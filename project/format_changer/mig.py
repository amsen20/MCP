import csv

rows = []

with open('MNIST.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for _, row in enumerate(spamreader):
        row = [float(it) for it in row]
        for i in range(len(row)-10):
            row[i] /= 255
        row = [str(it)[:9] for it in row]
        row = ' '.join(row)
        rows.append(row)
        if _ % 1000 == 0:
            print(_)

content = "\n".join(rows)
content += "\n"

with open("MNIST.data", "w") as f:
    f.write(content)
