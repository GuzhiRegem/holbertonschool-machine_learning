#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ["Farrah", "Fred", "Felicia"]
plt.bar(labels, fruit[0], color="r", width=0.5)
plt.bar(labels, fruit[1], color="yellow", width=0.5, bottom=np.sum(fruit[:1], axis=0))
plt.bar(labels, fruit[2], color="#ff8000", width=0.5, bottom=np.sum(fruit[:2], axis=0))
plt.bar(labels, fruit[3], color="#ffe5b4", width=0.5, bottom=np.sum(fruit[:3], axis=0))
plt.legend(["apples", "bananas", "oranges", "peaches"])
plt.ylim(0, 80)
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.show()
