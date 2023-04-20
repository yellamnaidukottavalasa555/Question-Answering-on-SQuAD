# Question-Answering-on-SQuAD-Dataset
This project aimed to explore the effectiveness of using the DistilBERT-base-uncased-distilled-squad, ALBERT-base-v2, Google/ELECTRA-base-generator models on the Stanford SQUAD dataset for question answering tasks. The objective was to achieve a high F1 score to demonstrate the potential impact of these models in natural language processing. The results of the DistilBERT-base-uncased-distilled-squad model showed a significant improvement in F1 score, with a final score of 84.19. These findings indicate that this model is highly effective for question answering tasks and have the potential to improve the accuracy of natural language processing systems.

# Work Flow

## Dataset
The Stanford Question Answering Dataset (SQuAD) is a popular benchmark dataset for evaluating the performance of question answering systems. The dataset consists of more than 100,000 question-answer pairs, covering a diverse range of topics such as science, history, and literature. Each question-answer pair is associated with a corresponding passage of text that provides the necessary context for answering the question.

# Models
1. ELECTRA Model (google/electra-base-generator)
2. ALBERT (albert-base-v2)
3. DistilBERT (distilbert_base_uncased_distilled_squad)

# Results
# Results
|                                             | Loss | Exact Score | F1 Score |
| :------------------------------------------:| :---:| :----------:| :-------:|
|      **google/electra-base-generator**      | 1.06 |    57.69    |   73.68  |
|              **albert-base-v2**             | 0.65 |    64.89    |   80.45  |
| **distilbert_base_uncased_distilled_squad** | 0.52 |    69.36    |   84.19  |

# Conclusion
Based on the analysis conducted, we concluded that the "distilbert-base-uncaseddistilled-squad" model is the most suitable for our task. This model achieved the lowest loss of 0.52 and the highest exact score of 69.36 and F1 score of 84.19 among the three models applied. It is also important to consider factors such as computational resources, training time, and ease of use when selecting a model. The "distilbert-base-uncased-distilled-squad" model may be the most effective in terms of accuracy, but it may also require more resources and longer training times than the other models.

Finally, it is important to emphasize that the results of this report are specific to the dataset and task that were used. Other datasets or tasks may require different models or parameter settings, and it is always important to evaluate the performance of a model on a new task or dataset before drawing any conclusions about its effectiveness.
