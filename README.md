# Implementação de Algoritmos de Deep Learning com PyTorch: Classificação do Dataset Iris

# Descrição do Projeto
Este mini-projeto implementa uma rede neural simples em PyTorch para classificar espécies de flores no dataset Iris. O projeto está dividido em dois notebooks principais: um para treinamento e salvamento dos pesos `train_model.ipynb` e outro para carregar os pesos e realizar predições `predict_model.ipynb`.

# Estrutura do Repositório

* `train_model.ipynb`: Arquivo para treinar o modelo, registrar métricas e salvar os melhores pesos.

* `predict_model.ipynb`: Arquivo para carregar os pesos treinados e fazer predições em novos dados.

* `best_model.pth`: Arquivo que contém os pesos da melhor rede treinada, salvo automaticamente durante o treinamento.

* `README.md`: Documento explicativo do projeto.


# Descrição dos Notebooks
## Notebook 1: `train_model.ipynb`
O notebook de treinamento contém as seguintes etapas:

1. Importação das Bibliotecas: Todas as bibliotecas necessárias são importadas, incluindo PyTorch para a construção da rede neural e bibliotecas do scikit-learn para pré-processamento e métricas de desempenho.

   ![image](https://github.com/user-attachments/assets/6591dd08-449f-4a6a-abb6-596abb2c280e)

2. Carregamento e Pré-Processamento dos Dados:
* O dataset Iris é carregado, e os dados são divididos em conjunto de treino e teste (80/20).
* Os dados são normalizados para melhorar a eficiência do treinamento.
* Os dados são convertidos para tensores PyTorch.

   ![image](https://github.com/user-attachments/assets/50e24572-2728-4a5b-8918-a55296add188)


3. Definição do Modelo:
* Um modelo de rede neural simples é definido com uma camada oculta de 16 neurônios e uma camada de saída com 3 neurônios, correspondente às 3 classes do dataset Iris.

   ![image](https://github.com/user-attachments/assets/170c4e63-e9d5-45a7-878a-15ce5dd292d6)


4. Configuração da Função de Perda e Otimizador:
* Utiliza-se a função de perda CrossEntropyLoss e o otimizador Adam com taxa de aprendizado de 0.001.

   ![image](https://github.com/user-attachments/assets/2d5758e1-466a-44fd-ba8d-f63d9a13db89)


5. Treinamento do Modelo com Registro de Métricas:
* O modelo é treinado por 50 épocas.
* Durante o treinamento, são registradas as seguintes métricas:
    * Loss de Treinamento e Validação
    * Acurácia de Treinamento e Validação
    * F1 Score de Validação
    * Precisão e Recall de Validação
* Os melhores pesos do modelo (baseado na menor loss de validação) são salvos em `best_model.pth`.

   ![image](https://github.com/user-attachments/assets/54ba69b2-4594-4aff-a626-d159c70496f3)

   ![image](https://github.com/user-attachments/assets/fca522b3-257e-4562-a062-ca8b209338b2)

6. Visualização das Métricas:
* São exibidos gráficos que mostram o desempenho do modelo ao longo do treinamento:
    * Loss de Treinamento e Validação
      
  ![image](https://github.com/user-attachments/assets/08ed9a62-7b52-4a45-8c06-4ddd80a0f74f)

    * Acurácia de Treinamento e Validação

  ![image](https://github.com/user-attachments/assets/d46ba931-f149-4e0b-98b3-3efa40ad1ddb)

    * F1 Score de Validação

  ![image](https://github.com/user-attachments/assets/f29efbdf-17fe-4b31-8773-e96dda9176fc)

    * Precisão e Recall de Validação
  
  ![image](https://github.com/user-attachments/assets/d36c4099-4a9f-4b2e-8ac9-3db71d3ec604)


* A matriz de confusão é exibida ao final para avaliar o desempenho do modelo em cada classe.

   ![image](https://github.com/user-attachments/assets/fce7f1ea-e9eb-4291-9c8e-cbadbf1a0e77)
  

<br>

## Notebook 2: `predict_model.ipynb`
O notebook de predição contém as seguintes etapas:

1. Importação das Bibliotecas:
* Importa as bibliotecas necessárias, incluindo PyTorch para carregar o modelo e fazer predições.

   ![image](https://github.com/user-attachments/assets/a9d98cbb-a11c-403a-be84-c1021ad86814)


2. Definição do Modelo:
* A mesma arquitetura de rede neural definida no notebook de treinamento é redefinida aqui, garantindo compatibilidade para carregar os pesos.

   ![image](https://github.com/user-attachments/assets/f369a889-a9da-4cd6-8592-a79944a5103f)


3. Carregamento dos Pesos:
* Os melhores pesos do modelo são carregados a partir do arquivo `best_model.pth`, colocando o modelo em modo de avaliação.

![image](https://github.com/user-attachments/assets/461e3da5-59c6-4b8f-b58b-9f398d26b09d)

![image](https://github.com/user-attachments/assets/372adc2d-289e-4351-b310-3fa9984073b7)


4. Predição em Novos Dados:
* Um exemplo de entrada é usado para demonstrar a capacidade do modelo de realizar predições.
* O modelo retorna a classe prevista para a amostra.

![image](https://github.com/user-attachments/assets/1d9714a5-adf9-450b-8647-bcdfe03fc381)

![image](https://github.com/user-attachments/assets/7c4be428-1a64-425e-9183-64de05f1a0e0)

# Principais Métricas de Desempenho
As seguintes métricas são registradas e exibidas para avaliar o desempenho do modelo:

* `Loss de Treinamento e Validação`: Ajuda a entender se o modelo está aprendendo ao longo das épocas.
* `Acurácia de Treinamento e Validação`: Mede a precisão geral das predições do modelo.
* `F1 Score de Validação`: Útil para avaliar o equilíbrio entre precisão e recall.
* `Precisão e Recall de Validação`: Oferecem uma visão detalhada sobre a performance do modelo em cada classe.
* `Matriz de Confusão`: Exibe onde o modelo acerta e erra para cada classe do conjunto de teste.

# Como Executar os Notebooks
## Requisitos
Certifique-se de que todas as bibliotecas necessárias estão instaladas.

## Passo 1: Treinamento
1. Baixe o arquivo `train_model.ipynb`.
2. Execute-o. Os melhores pesos serão salvos automaticamente como `best_model.pth`.

![image](https://github.com/user-attachments/assets/d947c9cd-f1b0-4a87-8fbf-17ad3493b4ee)


## Passo 2: Predição
1. Após o treinamento, abra o arquivo `predict_model.ipynb`.
2. Carregue os pesos salvos (fazendo upload do arquivo `best_model.pth` , inserindo o caminho dele ou buscando o mesmo na pasta 'Downloads' onde ele deve ter sido encaminhado após a execução do arquivo de treinamento) e execute as células para realizar predições em novos dados.

<br>

# Conclusão
Este projeto demonstra o uso de uma rede neural simples para classificação de dados usando PyTorch. Através de métricas como acurácia, F1 Score, precisão e recall, é possível avaliar detalhadamente o desempenho do modelo. Este projeto serve como uma introdução ao uso de redes neurais para tarefas de classificação em Python e PyTorch.

# Discentes: 
* Nathalia Ohana
* Daniel Marinho
* Felipe Ribeiro

