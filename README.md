# predictive-purchase-analytics
A Machine Learning approach to predict online shopping completion using K-Nearest Neighbors (k-NN). Features data preprocessing, model evaluation with sensitivity/specificity metrics, and performance analysis.

Este projeto utiliza o algoritmo K-Nearest Neighbors (k-NN) para prever se um utilizador irá finalizar uma compra online com base em variáveis de comportamento (tempo em páginas, taxa de rejeição, mês, etc.).

Diferente de modelos de classificação genéricos, aqui o foco é o equilíbrio entre:

Sensibilidade (Recall): A capacidade do modelo identificar quem realmente vai comprar.

Especificidade: A capacidade do modelo identificar quem não vai comprar.

Nota Técnica: Em e-commerce, muitas vezes preferimos um modelo que identifique potenciais compradores (alta sensibilidade) mesmo que tenhamos alguns falsos positivos, para fins de marketing direcionado.

## 🛠️ Tecnologias
Python 3.x

Scikit-Learn: Para o modelo k-NN e pré-processamento.

Pandas: Para manipulação do dataset.
