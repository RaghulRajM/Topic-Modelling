# Topic-Modelling

Hackathon Overview:

In this hackathon, participants are tasked with a challenging objective: identifying and categorizing themes within a given corpus of text data collected from diverse sources. The primary goal is to develop techniques that yield top-notch performance in accurately categorizing the provided text data. The dataset encompasses information from various domains, making the task intellectually stimulating and multi-faceted.

Dataset Labels/Topics:
1. "glassdoor_reviews"
2. "tech_news"
3. "room_rentals"
4. "sports_news"
5. "Automobiles"

**Evaluation Criteria:**
The performance metric for this competition is the accuracy score, a fundamental measure in classification tasks. The accuracy is computed using the formula:

\[ Accuracy = \frac{(TP + TN)}{(TP + FP + TN + FN)} \]

where TP (True Positive), TN (True Negative), FP (False Positive), and FN (False Negative) are components derived from the confusion matrix.

Sklearn Implementation:
```python
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]  # Predicted labels
y_true = [0, 1, 2, 3]  # True labels
accuracy = accuracy_score(y_true, y_pred)
```

**Approach:**
The participant shares insights into their approach, highlighting the utilization of conventional Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) techniques for theme extraction. However, a notable twist in this competition is its semi-supervised nature. Unlike typical unsupervised scenarios, here, the titles of the themes are known in advance.

**Innovation with Guided LDA:**
In response to the semi-supervised challenge, the participant describes their innovative solution – the incorporation of Guided LDA. Unlike its unsupervised counterparts, Guided LDA allows the integration of predefined "seed" words for each topic. These seed words act as guidance for the model, enhancing its accuracy in identifying topics. This approach reflects a thoughtful and effective workaround, demonstrating the participant's commitment to leveraging state-of-the-art methodologies for optimal results in a semi-supervised context.

This hackathon not only tests participants' technical prowess but also encourages creativity in addressing real-world challenges, showcasing the dynamic intersection of machine learning and problem-solving skills.
