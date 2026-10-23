# Java MLP Digit Classifier

Implementação em Java de uma rede neuronal MLP para classificação binária de dígitos manuscritos **2 e 3**, desenvolvida no contexto académico de Inteligência Artificial.

## Funcionalidades

- MLP implementado de raiz em Java
- Backpropagation e treino por mini-batch
- Normalização e divisão estratificada dos dados
- Regularização L2 e early stopping
- Data augmentation
- Ensemble de modelos
- Ajuste automático de threshold
- Operações matriciais e funções de ativação implementadas sem frameworks de machine learning

## Arquitetura

A rede principal usa uma topologia `400 → 16 → 8 → 1`, recebendo imagens 20×20 (400 atributos) e produzindo uma probabilidade usada para distinguir os dígitos 2 e 3.

## Estrutura principal

- `src/DigitTrainer.java` — treino, avaliação, holdout estratificado, augmentation e ensemble
- `src/MLP.java` — implementação da rede neuronal e backpropagation
- `src/Matrix.java` — operações matriciais
- `src/P4.java` — classificador final e carregamento dos modelos treinados
- `src/Sigmoid.java` — função de ativação sigmoid
- `src/Step.java` — função de decisão binária
- `src/pesos/` — pesos e threshold dos modelos treinados

## Tecnologias

Java · Machine Learning · Redes Neuronais · MLP · Backpropagation · Ensemble Learning

## Contexto

Projeto académico desenvolvido na Universidade do Algarve. O modelo inclui regularização, early stopping e ensemble para melhorar a generalização do classificador.