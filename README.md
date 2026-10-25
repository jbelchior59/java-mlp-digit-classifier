# Java MLP Digit Classifier

Implementação em Java de uma rede neuronal MLP para classificação binária de dígitos manuscritos **2 e 3**, desenvolvida no contexto académico de Inteligência Artificial.

## Funcionalidades

- MLP implementado de raiz em Java
- Backpropagation e treino por mini-batch
- Normalização e divisão estratificada dos dados
- Regularização L2 e early stopping
- Data augmentation
- Ensemble de 10 modelos
- Ajuste automático de threshold
- Operações matriciais e funções de ativação implementadas sem frameworks de machine learning

## Arquitetura

A rede principal usa uma topologia `400 → 16 → 8 → 1`, recebendo imagens 20×20 (400 atributos) e produzindo uma probabilidade usada para distinguir os dígitos 2 e 3.

## Conteúdo publicado

A versão completa do projeto está publicada na branch `develop`. Inclui:

- `src/DigitTrainer.java` — treino, validação, seleção de threshold e geração do ensemble
- `src/MLP.java` — implementação da rede neuronal e backpropagation
- `src/Matrix.java` — operações matriciais
- `src/P4.java` — classificador final e carregamento dos modelos treinados
- `src/MLPNXOR.java` — exemplo adicional de treino de uma MLP
- `src/IDifferentiableFunction.java`, `src/Sigmoid.java` e `src/Step.java` — funções e contratos auxiliares
- `src/pesos/` — pesos base, `threshold.txt` e os modelos `model_00` a `model_09`
- `bundle/` — partes usadas para reconstruir o pacote académico completo
- `setup-and-test.bat` — reconstrução, compilação e teste local automatizados no Windows

A branch `main` mantém uma versão mais pequena do código-fonte; para usar o conjunto completo acima, usa `develop`.

## Compilar e validar no Windows

É necessário ter um JDK instalado e `javac` disponível no `PATH`.

```bat
git checkout develop
setup-and-test.bat
```

O script reconstrói o pacote completo quando necessário, compila `complete/src/*.java` para `out/` e executa uma validação técnica do ensemble com uma entrada de 400 zeros. Se o classificador imprimir `2` ou `3`, os modelos e o threshold foram carregados corretamente.

## Executar o classificador

Depois de executar `setup-and-test.bat`, a partir da pasta `complete`:

```bat
java -cp ..\out P4 < input.txt
```

A entrada deve fornecer os 400 valores esperados por `P4` para uma imagem 20×20.

## Tecnologias

Java · Machine Learning · Redes Neuronais · MLP · Backpropagation · Ensemble Learning

## Contexto

Projeto académico desenvolvido na Universidade do Algarve. O treino final combina regularização, early stopping, data augmentation, seleção de threshold e ensemble para melhorar a generalização do classificador.
