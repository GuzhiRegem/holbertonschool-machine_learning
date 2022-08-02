#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

h_s = {"linewidth":1.0, "edgecolor":'k'}
plt.hist(student_grades, list(range(0, 110, 10)), **h_s)
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.show()
