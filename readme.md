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