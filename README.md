# XAI_Battery

Battery capacity degradation is a major challenge for modern energy storage technologies, influenced by mechanisms such as SEI formation, electrolyte breakdown, and lithium plating. While deep-learning approaches have gained interest for modeling degradation, they often underperform traditional statistical methods and lack consistency.

In this work, we apply explainable AI (XAI) techniques to better understand deep-learning behavior in battery degradation prediction. Using publicly available datasets, we employ the capacity matrix as a compact representation of electrochemical cycling data and train CNN and Transformer models to predict the remaining useful life (RUL) of batteries.

To interpret model predictions, we apply Grad-CAM, revealing attention patterns that provide insight into the decision-making process of the deep-learning models. This combination of predictive modeling and interpretability aims to improve reliability and understanding of RUL forecasts in battery systems.

Finally, by using a model-selection program during training, we are able to eliminate models that focus on the wrong parts of the data, resulting in a significant improvement in both consistency and performance.
| **Metric**             | **Normal Models** | **Guided Trained Models** | **Improvement** |
| ---------------------- | ----------------- | ------------------------- | --------------- |
| **Mean RMSE**          | 114.43            | 84.11                     | 26.50%          |
| **Standard Deviation** | 83.03             | 4.34                      | 94.77%          |

| **Metric**             | **Normal Models** | **Guided Trained Models** | **Improvement** |
| ---------------------- | ----------------- | ------------------------- | --------------- |
| **Mean RMSE**          | 84.23             | 82.07                     | 2.56%           |
| **Standard Deviation** | 6.05              | 5.63                      | 6.94%           |

| **Metric**             | **Normal Models** | **Guided Trained Models** | **Improvement** |
| ---------------------- | ----------------- | ------------------------- | --------------- |
| **Mean RMSE**          | 106.17            | 102.93                    | 3.05%           |
| **Standard Deviation** | 17.94             | 13.57                     | 24.36%          |
