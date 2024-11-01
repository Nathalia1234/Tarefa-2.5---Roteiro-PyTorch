# Implementação de Algoritmos de Deep Learning com PyTorch: Classificação do Dataset Iris

# Descrição do Projeto
Este mini-projeto utiliza o dataset clássico Iris para treinar um modelo de rede neural em PyTorch, visando classificar espécies de íris. O objetivo é demonstrar a aplicação de técnicas de Deep Learning para resolver problemas de classificação. Este repositório inclui todos os componentes necessários para entender, treinar, avaliar, e executar o modelo.

# Estrutura do Repositório
train_model.ipynb: Notebook para treinamento do modelo, incluindo etapas de pré-processamento, configuração da rede neural e visualização das métricas.
predict_model.ipynb: Notebook para carregamento dos pesos treinados e predições com novos dados.
best_model.pth: Arquivo contendo os pesos da melhor versão do modelo treinado.
README.md: Documento explicativo sobre o projeto, descrições dos dados e as decisões tomadas.
Vídeo: Um vídeo curto que demonstra a execução e funcionamento do modelo.

# Dataset
O dataset utilizado é o Iris, composto por quatro características (comprimento e largura da sépala e pétala) para três espécies de flores. Ele possui 150 amostras, distribuídas uniformemente entre as classes.

## Pré-Processamento dos Dados
Codificação das Classes: Usamos o LabelEncoder para transformar as classes categóricas em valores numéricos.
Normalização: Aplicamos normalização para melhorar a eficiência do treinamento e a estabilidade dos gradientes.
Divisão em Conjuntos: Dividimos o dataset em conjuntos de treino e teste com a proporção 80/20.

## Estrutura da Rede Neural
Optamos por uma rede simples com uma camada oculta, suficiente para capturar a complexidade do dataset Iris, que tem baixo número de amostras e dimensões:

* Camada de Entrada: 4 neurônios (um para cada característica).
* Camada Oculta: 16 neurônios com função de ativação ReLU.
* Camada de Saída: 3 neurônios (um para cada classe), ativados com Softmax durante a inferência para gerar probabilidades de cada classe.

A escolha por uma rede simples se justifica pela baixa complexidade do problema, permitindo um treinamento rápido e resultados satisfatórios.

# Treinamento
O modelo foi treinado por 50 épocas com Adam como otimizador e uma taxa de aprendizado de 0.001. Utilizamos CrossEntropyLoss como função de perda, ideal para classificação multiclasse.

Durante o treinamento, registramos as métricas principais para avaliar o progresso do modelo. Os pesos da melhor época foram salvos automaticamente.

# Avaliação de Desempenho
Para avaliar o desempenho, analisamos as seguintes métricas:

1. Acurácia: Medimos a acurácia do modelo em treino e teste.
2. F1 Score: Utilizado para avaliar a performance com equilíbrio entre precisão e recall.
3. Loss: Visualizamos a loss durante o treinamento e validação para garantir que o modelo está convergindo.
4. Matriz de Confusão: Demonstramos o desempenho do modelo para cada classe, ajudando a identificar potenciais classes mais difíceis de prever.

## Gráficos
Os gráficos de loss e acurácia por época e a matriz de confusão foram incluídos para facilitar a análise visual da performance do modelo.

# Instruções para Predições
O notebook predict_model.ipynb permite carregar os pesos treinados (best_model.pth) e fazer predições em novas amostras. Este arquivo demonstra como o modelo pode ser reutilizado para inferências futuras.

# Vídeo Demonstrativo
Incluímos um vídeo que mostra a execução do modelo, destacando o processo de predição e resultados para as amostras de teste.

# Conclusão
Este mini-projeto demonstrou a viabilidade de redes neurais simples para tarefas de classificação em datasets de baixo volume e baixa complexidade. A aplicação das técnicas de Deep Learning mostrou-se eficaz, e os resultados são promissores para a classificação de espécies de íris.

# Discentes: 
* Nathalia Ohana
* Daniel Marinho
* Felipe Ribeiro

