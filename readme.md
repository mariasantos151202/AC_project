# **CNN Arquitecture** 

Porque é que acahamos que o CNN é um bom modelo?

As redes neuronais convolucionais são ótimas para processamento de imagem e os sinais podem ser transformados em imagens através de transformações matemáticas (espectogramas, fonocardiogramas, ect), logo estas redes conseguem identificar bem os padrões nas frequencias. Uma vez que estamos a ter em conta 2 eixos: tempo e frequencia do sinal, faz sentido que seja um CNN 2D.


- a **primeira camada** é para aprender padrões mais simples, com 16 filtros de 3 por 3, e depois maxpooling2D, que reduz para metade.
- a **segunda camada** é para combinar os padrões simpes em padrões mais completos e formas maiores, com 32 filtros de 3 x 3, novamente com maxpooling2D.
- a **terceira camada** é para identificar os padrões mais abstratos, que já se parecem com o murmur, aplicando 64 filtros de 3 por 3 e maxpooling outra vez.
- a **quarta camada** que é a dense, recebe os 64 numeros que saiem do **Globar Average Pooling** (que faz a média de cada mapa e fica com um único número por mapa) com ativação relu
- **camada de output** -->  é a ultima camada dense, e faz a decisão e calcula a probabilidade das 3 classes (n_classes) com o softmax. 

                            x = layers.GlobalAveragePooling2D()(x)
                                x = layers.Dense(64, activation="relu")(x)
                                x = layers.Dropout(0.5)(x)
                                out = layers.Dense(n_classes, activation="softmax")(x)


### **Porque**?

1. Campo recetivo --> com 3 convoluções 3×3 e pooling 2×2 entre elas, cada neurónio da última camada vê cerca de 18 frames no eixo do tempo. Com frame_step=32 ms e frame_len=128 ms, isso são (18−1)×32+128 ≈ 670 ms. É da ordem de grandeza de um ciclo cardíaco (S1, sístole, S2), que é onde o sopro aparece.
Cálculo exato do Campo Recetivo ($RF$): 
fórmula clássica $RF_{l} = RF_{l-1} + (k - 1) \times S_{l-1}$, onde $S$ é o stride cumulativo
- Camada 1: Conv $3\times3$ (stride 1) $\rightarrow RF_1 = 3$
    - Pool 1: MaxPool $2\times2$ (stride 2) $\rightarrow RF = 3 + (2 - 1)\times 1 = 4$; Stride acumulado $S = 2$
- Camada 2: Conv $3\times3$ (stride 1) $\rightarrow RF = 4 + (3 - 1)\times 2 = 8$
    - Pool 2: MaxPool $2\times2$ (stride 2) $\rightarrow RF = 8 + (2 - 1)\times 2 = 10$; Stride acumulado $S = 4$
- Camada 3: Conv $3\times3$ (stride 1) $\rightarrow RF = 10 + (3 - 1)\times 4 = 18$
    - Pool 3: MaxPool $2\times2$ (stride 2) $\rightarrow RF = 18 + (2 - 1)\times 4 = 22$

Ou seja: antes do último pooling são 18 frames, e após o 3.º pooling são 22 frames, o que coincide com um ciclo cardiaco --> ideal !

2. Poucos parâmetros ---> a rede tem cerca de 28 mil parâmetros. Com poucos doentes e gravações muito ruidosas, uma rede grande tenderia a overfitting. O Dropout (0,25 nos blocos, 0,5 na cabeça) vai no mesmo sentido --> desativa os neurónios aleatoriamente em cada passo de treino, impede que a rede dependa de padrões específicos e obriga-a a aprender features mais robustas. O BatchNorm estabiliza o treino e tem um efeito regularizador ligeiro. 

3. 16→32→64 filtros é o padrão habitual na literatura: à medida que o pooling reduz a resolução ( para metade cada vez que existe pooling), aumenta-se o número de canais para manter a capacidade de representação.

3. GlobalAveragePooling em vez de Flatten --> permite que haja redução de parâmetros e dá invariância temporal, que é útil porque o sopro pode estar em qualquer ponto da janela. Também torna a rede independente do tamanho da entrada.


### **Resultados**

- criei dados sintéticos enquanto os nossos dados ainda estavam a sofrer o preprocessamento para ver efetivamente como a rede funcionava e o output que era produzido: