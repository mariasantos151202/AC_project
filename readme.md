Projeto de AC 2

então a ideia será criar duas formas diferentes de processamento, mas apenas na parte final.

A maior parte do pré processamento será comum aos dois modelos, ou seja, reamostrar, filtrar, normalizar, cortar em janelas, etc., será feita por igual para tudo.

Apenas as partes finais mudam, estas sendo: 
-- CNN 2D pede uma entrada tipo imagem: um espectrograma/log-mel de forma (tempo, frequência, 1). Uma CNN 1D pode usar o sinal em bruto, de forma (amostras, 1).
-- RNN pede uma sequência: (passos_de_tempo, nº_de_features). O log-mel serve, só que cada frame passa a ser um passo temporal em vez de uma coluna da imagem.

Apenas a extração de features vai diferir.

## **Explicações**
**Porquê cortar tudo a baixo dos 25 Hz?** Abaixo disso há quase só lixo, como o movimento do estetoscópio na pele, a respiração e o ruído de manuseamento. É uma deriva lenta de baixa frequência (baseline wander) que não tem informação cardíaca útil, mas tem muita amplitude e domina a normalização. O que pode influenciar com os resultados

**Porquê cortar acima dos 400 Hz?** os sons cardíacos normais (S1 e S2) concentram a energia aproximadamente entre 20 e 200 Hz. Os sopros são mais “ruidosos” e estendem-se mais acima, até algumas centenas de Hz. Acima disso há pouca energia cardíaca e muito ruído (fricção, ambiente, eletrónica, vozes).

*Nota: Enquanto 25-400 Hz é uma escolha comum (a maior parte das soluções no challenge usam estes valores) não é algo 100% certo. Alguns sopros têm componentos que chegam a mais ou menos 600 Hz, por isso ao cortarmos a 400 Hz podemos estar a perder informação. Será uma escolha que temos que fazer. Podemos escolher 25-400 Hz, mas vamos ter de justificar, a minha sugestão é testarmos outros valores (tipo 20 a 500 Hz), isso vai nos dar mais pontos (muito provavelmente)*

**O que é o Nyquist?** Refere-se ao Teorema de Nyquist e é importante por causa da forma que fazemos a mudança de frequência das coisas, segue aqui a definição: O teorema de Nyquist (ou teorema da amostragem de Nyquist–Shannon) define que, para converter um sinal contínuo (analógico) em sinal digital sem perda de informação, a frequência de amostragem deve ser pelo menos duas vezes maior que a frequência máxima do sinal

**Porquê normalizar (isto é mais para mim)** 
1. A amplitude de uma gravação depende de coisas que não têm a ver com o sopro: a pressão do estetoscópio, o ganho do microfone, a espessura do tórax, o tamanho da criança, o ruído do ambiente. Sem normalizar, a rede pode aprender atalhos como “gravação com amplitude alta = Present”, que não generalizam para pacientes novos. Com o z-score, o modelo tem de olhar para a forma do sinal (o padrão no tempo e na frequência) e não para o volume.
2. Remover o offset (média). Algumas gravações têm um valor médio diferente de zero (componente DC). Subtrair a média centra o sinal em zero. Aqui o passa-banda já remove a componente DC (está abaixo dos 25 Hz), mas o z-score garante-o na mesma.
3. As redes neuronais treinam melhor quando as entradas têm uma escala previsível (à volta de -3 a 3). Com valores muito grandes ou muito pequenos, os gradientes ficam mal condicionados, o otimizador (Adam) tem de se adaptar à escala, e as inicializações dos pesos (pensadas para entradas de variância ~1) deixam de ser adequadas. Isto é especialmente importante nas RNN, em que valores grandes se propagam ao longo de muitos passos temporais.
4. Todos os exemplos passam a ter a mesma escala, por isso o erro de um paciente não domina a loss só por ter um sinal mais forte.