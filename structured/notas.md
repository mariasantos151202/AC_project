# **Convolutional Neural Networks**

Para explicar a escolha da nossa arquitetura CNN 2D, partimos primeiro da física e da biologia do sinal cardíaco, e não de um número arbitrário de parâmetros. Como um ciclo cardíaco completo dura entre 670 e 800 milissegundos, precisávamos que os filtros da última camada conseguissem observar a sístole e a diástole em simultâneo para distinguir um sopro contínuo de um ruído isolado. Com apenas um ou dois blocos convolucionais, a rede só veria fragmentos curtos do batimento; ao estruturarmos três blocos consecutivos ( 16→32→64 filtros, 3×3 com MaxPooling 2×2), o campo recetivo expande-se exatamente o suficiente para cobrir um ciclo cardíaco inteiro sem perder resolução temporal.

Uma vez definida essa profundidade de três blocos pela natureza biológica do problema, tivemos de garantir que a rede não ficava pesada demais para a quantidade de dados que temos no dataset CirCor (cerca de 16 600 janelas de treino). Se tivéssemos fechado a rede com uma camada densa tradicional após um Flatten, o modelo saltaria para centenas de milhares de parâmetros e entraria rapidamente em overfitting, decorando o ruído de fundo específico de cada paciente. Para evitar isso, substituímos o Flatten por GlobalAveragePooling2D, o que manteve a arquitetura completa em apenas cerca de 28 000 parâmetros.

Na prática, isto dá-nos uma relação muito equilibrada entre dados e complexidade — cerca de 1,7 parâmetros treináveis por janela de treino. Como a rede não tem "memória" suficiente para decorar exemplos individuais, é obrigada a aprender padrões acústicos gerais e consistentes entre pacientes. Este equilíbrio explica por que razão a nossa CNN compacta desenhada de raiz atingiu um Weighted Accuracy de validação de 0,83, superando claramente modelos genéricos muito maiores como a ResNet50 (0,71), cujos milhões de filtros pré-treinados em fotografias não se adaptam tão bem à física de espectrogramas cardíacos.

**Justificação da loss:**
A WA (a métrica do enunciado) não é diferenciável, por isso não se pode usar diretamente como loss. A cross-entropy é o substituto razoável, e a WA fica como métrica de seleção.


### **Sobre o cnn.py**

- exemplo de implementação de uma cnn ( o claude sugeriu e eu estive a ver o que era tudo e fez sentido, a explicação do que é tudo está a seguir ). 
- Também tem uma função para gerar dados artificais para conseguirmos ver alguns resultados. Até agora parece estar tudo bem para os dados artificais gerados, com uma accuracy elevada. 
- Os nossos dados têm imbalance de classes portanto depois vamos ter de alterar algumas coisas quando testarmos com os nossos dados !

Tive a pesquisar e parece me que 3 blocos convolucionais e 2 densos são um bom ponto de partida: dão profundidade suficiente para aprender padrões em vários níveis, cabem no tamanho do espectrograma (40 x 153), estou a assumir que o joão vai transformar em espetogramas, e mantêm o modelo pequeno para um dataset limitado.

- a **primeira camada** é para aprender padrões mais simples, com 16 filtros de 3 por 3, e depois maxpooling2D, que reduz para metade.
- a **segunda camada** é para combinar os padrões simpes em padrões mais completos e formas maiores, com 32 filtros de 3 x 3, novamente com maxpooling2D.
- a **terceira camada** é para identificar os padrões mais abstratos, que já se parecem com o murmur, aplicando 64 filtros de 3 por 3 e maxpooling outra vez.
- a **quarta camada** que é a dense, recebe os 64 numeros que saiem do **Globar Average Pooling** (que faz a média de cada mapa e fica com um único número por mapa) com ativação relu
- **camada de output** -->  é a ultima camada dense, e faz a decisão e calcula a probabilidade das 3 classes (n_classes) com o softmax. 

                            x = layers.GlobalAveragePooling2D()(x)
                                x = layers.Dense(64, activation="relu")(x)
                                x = layers.Dropout(0.5)(x)
                                out = layers.Dense(n_classes, activation="softmax")(x)




























### **Sobre os diferentes parametros de uma CNN 2D: exemplo do cnn.py, que já esta mais ou menos adaptado ao nosso contexto**

**Forma dos dados** (n, 64, 128, 1): n exemplos, cada um com 64 linhas (frequências), 128 colunas (tempo) e 1 canal. O None que aparece no model.summary() é o número de exemplos por batch, que varia.

**Conv2D(16, 3)** --> número de filtros e tamanho: 16 filtros de 3×3 que deslizam pela imagem. Cada filtro aprende a detetar um padrão (por exemplo, “uma faixa de energia horizontal”). Mais filtros, mais padrões diferentes.

**padding="same": o que é padding?** --> acrescentar uma moldura de 0 à volta da imagem antes de aplicar a convolução, o que permite que o filtro capte os cantos das imagens sem encolher no final. o same é para indicar que o tamanho inicial se vai manter no fim.

**activation="relu"** --> função de ativação que já sabemos como funciona: função que põe a zero os valores negativos e deixa passar os positivos. Sem uma função assim, empilhar camadas seria equivalente a uma só camada simples.

**MaxPooling2D(2)**: divide a imagem em quadrados 2×2 e guarda só o maior valor de cada um. A imagem fica com metade do tamanho (64×128 → 32×64) e a rede fica mais leve e menos sensível à posição exata dos padrões.

**Dropout(x)**: durante o treino, desliga x% , pode se 25 ou 50, das unidades ao acaso em cada passo. Obriga a rede a não depender de poucas unidades, o que reduz o overfitting. Em previsão não faz nada, ou seja atua apenas no treino.

**BatchNormalization**: normaliza os valores a meio da rede, o que estabiliza e acelera o treino. Funciona de forma diferente no treino e no teste. Normaliza os valores antes de passarem pela função de ativação.


**GlobalAveragePooling2D**: para cada um dos 64 mapas de features, faz a média de todos os valores e fica com 1 número. Passa-se de (8, 16, 64) para 64 valores, o que tem muito menos parâmetros do que esticar tudo (Flatten) e reduz o sobreajuste.

**Dense(64)**: camada “normal” em que cada unidade vê todos os 64 números anteriores.

**softmax**: transforma os 3 números finais em probabilidades que somam 1. A classe com maior probabilidade é a previsão.

**n_classes=3**: Absent, Unknown, Present. No nosso caso, 0, 1 e 2. Passado como argumento na inicialização do modelo



### Cenas sobre o batch normalization e o porque de ser importante para obtermos melhores resultados:

- Batch normalization is a technique that normalizes the activations of a layer within a mini-batch during the training of deep neural networks.

- It operates by calculating the mean and variance of the activations for each feature in the mini-batch and then normalizing the activations using these statistics.

- The normalized activations are then scaled and shifted using learnable parameters, allowing the model to adapt to the optimal activation distribution.



**what is flatten ??**

The flatten layer in a Convolutional Neural Network (CNN) is a crucial component that transforms multi-dimensional output from convolutional or pooling layers into a single, long, continuous vector.


**O que é a variancia temporal e espacial** de um sopro (nosso contexto)?
- Porque temporal? O fenómeno que estamos a trabalhar refere-se ao momento em que o sopro acontece ao longo da gravação ("em que ponto da janela ele ocorre").O eixo horizontal da nossa entrada 2D (espectrograma) representa o tempo. Logo, deslocar o sopro de uma posição temporal para outra é uma translação temporal. 

Onde entra a confusão com "espacial"?
- Na literatura geral de deep learning para visão computacional, o GAP2D é tradicionalmente descrito como conferindo "invariância espacial" (spatial translation invariance), porque numa fotografia comum os eixos $(X, Y)$ são puramente espaciais (largura e altura).Quando convertes áudio num espectrograma 2D, a rede trata matematicamente a matriz como um espaço 2D, mas fisicamente os eixos têm significados diferentes:Eixo horizontal: Tempo (dimensão temporal).Eixo vertical: Frequência (dimensão espetral).




----------------------------------------------------------------------------------------------