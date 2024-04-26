import matplotlib.pyplot as plt

# Evaluation metrics for each model
models = ['Multinomial Naive Bayes', 'Logistic Regression', 'Support Vector Machine']
accuracy = [0.858, 0.76, 0.688]
precision = [0.834, 1.0, 1.0]
recall = [0.961, 0.612, 0.496]
f1_score = [0.893, 0.76, 0.663]

# Plotting
x = range(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, accuracy, width, label='Accuracy')
rects2 = ax.bar([i + width for i in x], precision, width, label='Precision')
rects3 = ax.bar([i + width*2 for i in x], recall, width, label='Recall')
rects4 = ax.bar([i + width*3 for i in x], f1_score, width, label='F1 Score')

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Evaluation Metrics for Each Model')
ax.set_xticks([i + 1.5 * width for i in x])
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()
