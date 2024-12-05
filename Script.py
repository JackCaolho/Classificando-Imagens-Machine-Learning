# Importação das bibliotecas necessárias
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# 1. Carregamento do dataset
# ------------------------------------------------------------
# O Fashion MNIST é um dataset contendo imagens de roupas (28x28 pixels) classificadas em 10 categorias.
dataset = keras.datasets.fashion_mnist
(imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste) = dataset.load_data()

# ------------------------------------------------------------
# 2. Exploração inicial dos dados
# ------------------------------------------------------------
# Verificando o tamanho e as dimensões dos dados
print(f"Tamanho do conjunto de treino: {len(imagens_treino)}")
print(f"Formato das imagens de treino: {imagens_treino.shape}")
print(f"Formato das imagens de teste: {imagens_teste.shape}")
print(f"Tamanho do conjunto de teste: {len(identificacoes_teste)}")

# Verificando o intervalo de valores das classes
print(f"Menor identificação de classe: {identificacoes_treino.min()}")
print(f"Maior identificação de classe: {identificacoes_treino.max()}")

# ------------------------------------------------------------
# 3. Exibição de exemplos do dataset
# ------------------------------------------------------------
# As categorias correspondem às seguintes classes de roupas
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco', 
                           'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# Exibindo a primeira imagem de treino com uma barra de cores
plt.imshow(imagens_treino[0])
plt.colorbar()
plt.title(f"Classe: {nomes_de_classificacoes[identificacoes_treino[0]]}")
plt.show()

# ------------------------------------------------------------
# 4. Normalização das imagens
# ------------------------------------------------------------
# Normalizando os valores dos pixels para o intervalo [0, 1]
imagens_treino = imagens_treino / 255.0
imagens_teste = imagens_teste / 255.0

# ------------------------------------------------------------
# 5. Criação e configuração do modelo
# ------------------------------------------------------------
# Estrutura do modelo:
# - Flatten: transforma a imagem 28x28 em um vetor unidimensional.
# - Dense (256 neurônios): camada totalmente conectada com ativação ReLU.
# - Dropout (20%): para prevenir overfitting.
# - Dense (10 neurônios): camada de saída com ativação Softmax para classificação.
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Configurando o otimizador Adam com taxa de aprendizado de 0.002
adam = keras.optimizers.Adam(learning_rate=0.002)

# Adicionando callbacks:
# - EarlyStopping: para parar o treinamento cedo se a perda de validação não melhorar.
# - ModelCheckpoint: salva o melhor modelo com base na validação.
parando_cedo = [
    keras.callbacks.EarlyStopping(monitor='val_loss'),
    keras.callbacks.ModelCheckpoint(filepath='melhor_modelo.hdf5', monitor='val_loss', save_best_only=True)
]

# Compilando o modelo
modelo.compile(optimizer=adam,
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# ------------------------------------------------------------
# 6. Treinamento do modelo
# ------------------------------------------------------------
# Treinando o modelo por 5 épocas com 20% dos dados reservados para validação
historico = modelo.fit(imagens_treino, identificacoes_treino,
                       batch_size=480, epochs=5, validation_split=0.2,
                       callbacks=parando_cedo)

# ------------------------------------------------------------
# 7. Visualizando o histórico de treinamento
# ------------------------------------------------------------
# Plotando a acurácia ao longo das épocas
plt.plot(historico.history['accuracy'], label='Treino')
plt.plot(historico.history['val_accuracy'], label='Validação')
plt.title('Acurácia por épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Plotando a perda ao longo das épocas
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Perda por épocas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# ------------------------------------------------------------
# 8. Avaliação do modelo no conjunto de teste
# ------------------------------------------------------------
# Avaliando a perda e a acurácia no conjunto de teste
perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print(f"Perda no teste: {perda_teste}")
print(f"Acurácia no teste: {acuracia_teste}")

# ------------------------------------------------------------
# 9. Salvando e carregando o modelo treinado
# ------------------------------------------------------------
# Salvando o modelo no formato HDF5
modelo.save('modelo.h5')

# Carregando o modelo salvo
modelo_salvo = load_model('modelo.h5')

# ------------------------------------------------------------
# 10. Testando o modelo e o modelo salvo
# ------------------------------------------------------------
# Fazendo previsões com o modelo original
testes = modelo.predict(imagens_teste)
print(f"Previsão para a segunda imagem: {np.argmax(testes[1])}")
print(f"Classe real: {identificacoes_teste[1]}")

# Fazendo previsões com o modelo salvo
testes_modelo_salvo = modelo_salvo.predict(imagens_teste)
print(f"Previsão (modelo salvo) para a segunda imagem: {np.argmax(testes_modelo_salvo[1])}")
print(f"Classe real: {identificacoes_teste[1]}")
