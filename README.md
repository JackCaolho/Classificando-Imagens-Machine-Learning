# Classificação de imagens usando Machine Learning

Este repositório contém um projeto de machine learning desenvolvido em Python utilizando o TensorFlow/Keras para a classificação de imagens do dataset **Fashion MNIST**. O objetivo é criar um modelo capaz de identificar itens de vestuário com base em imagens em tons de cinza de 28x28 pixels.

## Objetivo do Projeto

O projeto visa treinar um modelo de rede neural para classificar imagens do **Fashion MNIST**, que contém 10 categorias de itens de vestuário, como camisetas, calças, bolsas, entre outros. O pipeline inclui carregamento, exploração, normalização dos dados, construção e treinamento do modelo, e avaliação de desempenho.

---

## Estrutura do Projeto

1. **Carregamento do Dataset**:
   - O dataset **Fashion MNIST** foi carregado diretamente da biblioteca `keras.datasets`.

2. **Exploração dos Dados**:
   - Verificamos dimensões das imagens, quantidade de amostras, e intervalo dos valores dos pixels.
   - Exibimos algumas amostras do dataset utilizando o `matplotlib`.

3. **Pré-processamento**:
   - Normalizamos os valores dos pixels dividindo por 255 para trabalhar com valores entre 0 e 1.

4. **Criação e Treinamento do Modelo**:
   - A rede neural é composta por:
     - Camada Flatten para converter a matriz 28x28 em um vetor unidimensional.
     - Camada Dense com 256 neurônios e ativação ReLU.
     - Camada Dropout para regularização e evitar overfitting.
     - Camada de saída Dense com 10 neurônios e ativação Softmax.
   - Utilizamos o otimizador **Adam** com uma taxa de aprendizado de 0.002.
   - Métrica utilizada: Acurácia.
   - Configuramos callbacks para Early Stopping e salvamento do melhor modelo.

5. **Visualização de Resultados**:
   - Plotamos gráficos de acurácia e perda durante o treinamento e validação.

6. **Avaliação e Testes**:
   - Avaliamos o modelo treinado nos dados de teste.
   - Realizamos previsões utilizando tanto o modelo treinado quanto o modelo salvo.

7. **Salvamento e Carregamento do Modelo**:
   - Salvamos o modelo treinado em um arquivo `.h5` para reutilização futura.

---

## Resultados

- O modelo alcançou uma acurácia de treino superior a 87% e validação de 88% após 5 épocas.
- A visualização dos gráficos mostrou uma boa convergência entre treino e validação, indicando um modelo bem generalizado.

---

## Como Utilizar

### Pré-requisitos
- Python 3.7 ou superior
- Bibliotecas: `tensorflow`, `matplotlib`, `numpy`

### Rodando o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/JackCaolho/machine-learning-classificando-imagens.git
