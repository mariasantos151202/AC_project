# Pré-processamento dos dados

---
---
## 1. Objetivo
Transformar as gravações de aúdio em bruto em tensores de tamanho fixo que possam ser usados por diferentes modelos, neste caso CNN e RNN. O pipeline tem duas etapas principais:

1. **Etapa comum** a ambos os modelos: leitura, reamostragem, filtragem, normalização e divisão em janelas.
2. **Etapa específica** a cada modelos: Extração de features (log-mel para CNN, MFCC para os RNN).

Assim, os dois modelos são treinados exatamente com as mesmas janelas e os mesmos pacientes, desta forma assegurando que podemos fazer uma comparação justa entre os dois.

```
.wav  ->  reamostrar  ->  passa-banda  ->  z-score  ->  janelas (5 s)
                                                            |
                              +-----------------------------------------------------------+
                              |                                                           |
                         log-mel (CNN 2D)                                           MFCC + deltas (RNN)
                       (N, 153, 40, 1)                                                  (N, 77, 39)
```
## 2. Dados e Etiquetas
- Dataset: CirCor DigiScope, parte pública (3163 gravações, 942 pacientes).
- Etiqueta usada: `Murmur` ∈ {Present, Unknown, Absent}, codificada como **0 = Present, 1 = Unknown, 2 = Absent**.
- A etiqueta é do **paciente**, não da gravação. Cada paciente tem várias gravações (uma por local de auscultação: AV, PV, TV, MV), e todas herdam a etiqueta do paciente.
- Distribuição das classes (pacientes): **[ 54,3% de Present / 35,9% Unknown / 9,8% Absent]**. As classes estão desequilibradas.

## 3. Divisão Treino/Validação/Teste
- Divisão **por paciente** (todas as gravações de um paciente ficam num só subjconjunto). Se não fosse assim, o modelos podia reconhecer o paciente em vvez de aprender o sopro o que iria fazer com que os resultados fossem otimistas.
- **Estratificada** pela class `Murmur` para manter proporções semelhantes em todos os subconjuntos.
- Proporções: 70% para treino, 15% para validação e 15% para teste, com semente fixa para garantir a reproducibalididade (seed=42)
- O resultado é então guardado num ficheiro de nome splits.json.
- São feitos **asserts** para garantir que não há pacientes repetidos nos subconjuntos.

| Subconjunto | Pacientes | Janelas |
|---|---|---|
| Treino | 658 | 16688 |
| Validação | 142 | 3644 |
| Teste | 142 | 3866 |

## 4. Pré-Processamento Comum
Parâmetros (classe `Config`):
 
| Parâmetro | Valor | Significado |
|---|---|---|
| `fs_target` | 1000 Hz | frequência de amostragem após reamostragem |
| `band` | (25, 400) Hz | banda do filtro passa-banda |
| `win_sec` | 5.0 s | tamanho da janela |
| `hop_sec` | 2.5 s | passo entre janelas (50% de sobreposição) |

### 4.1 Reamostragem (4000 → 1000 Hz)

A gravação original está feita a 4000 Hz. Pelo teorema de Nyquist, uma taxa *fs* só representa frequências até *fs/2*; a 1000 Hz o limite é 500 Hz, o que cobre a banda de interesse (até 400 Hz). Isto reduz o número de amotras 4 vezes, com menos memória e treino mais rápido. Usa-se `resample_poly`, que aplica um filtro anti-aliasing antes de baixar a taxa.

**Nota:** O teorema de Nyquist (ou teorema da amostragem de Nyquist–Shannon) define que, para converter um sinal contínuo (analógico) em sinal digital sem perda de informação, a frequência de amostragem deve ser pelo menos duas vezes maior que a frequência máxima do sinal.

### 4.2 Filtro passa-banda (25-400 Hz)

- **Abaixo dos 25 Hz:** Deriva de baixa frequência (movimento do estetóscopio, respiração, manuseamento do material), sem informação cardiaca útil mas com uma grande amplitude.
- **Acimaa ds 400 Hz:** Os sons cardíacos normais (S1, S2) concentram-se aproximadamente entre 20 a 200 Hz e os sopros estendem-se mais acima. Por isso, todo acima de 400 Hz é, na sua grande maioria, ruído ambiente com pouca energia cardíaca.
- Filtro **Butterworth de ordem 4**, com resposta plana na banda de passagem.
- Aplicado com `sosfiltfilt` (para a frente e para trás), o que dá **fase zero**: o sinal não é deslocado no tempo, o que é importante porque a posição de S1/S2 e do sopro entre eles contém informação. Usa-se a forma `sos` (secções de 2.ª ordem) por ser numericamente mais estável.
- A escolha 25–400 Hz é comum na literatura mas **não é uma lei**; alguns sopros têm componentes até ~600 Hz.

### 4.3 Normalização Z-Score (por gravação)

```
z = (x - média) / desvio_padrão
```

Cada gravação fica com média 0 e desvio-padrão 1. Remove diferenças de amplitude que não estão relacionadas com o sopro (pressão do estetoscopio, ganho, tamanho do paciente) e estabiliza o treino. É feita **depois** do filtro para que o que resta do sinal fique com desvio-padrão 1. Como é feita gravação a gravação não usa informação de outros pacientes, desta forma previne **fuga de dados** entre treino e teste.

*Limitação:* perde-se a amplitude absoluta (a intensidade do sopro é clinicamente relevante) e é sensível a picos (artefactos grandes comprimem o resto do sinal).

### 4.4 Janelas de Tamanho Fixo

- Janelas de 5 segundos com passo de 2,5 segundos. Gravações mais curtas que 5 segundos são completadas com zeros (*padding*).
- Razões: Uniformizar a entrada (as gravações têm entre 4,8 segundos e 80,4 segundos), ter mais exemplos de treino (funciona como data augmentation) e capturar vários batimentos (5 segundos contém entre 6 a 10 ciclos cardíacos).
- Cada janela herda o rótulo do paciente.
*Limitação:* um sopro pode ser audível só em alguns locais ou de forma intermitente, por isso algumas janelas "Present" podem não conter sopro (ruído nos rótulos). É isto que justifica a **agregação por paciente** na avaliação.
- Para cada janela guarda-se também o `pid`, o local (`loc`) e a gravação (`rec`) de origem.

## 5. Etapa 2: Features por Modelo

| Modelo | Função | Forma de saída |
|---|---|---|
| CNN 2D | `features_cnn2d` | `(N, 153, 40, 1)` |
| RNN | `features_rnn` | `(N, 77, 39)` |

### 5.1 Log-mel (CNN 2D)

1. **STFT** com janela de 128 amostras (128 ms) e passo de 32 amostras (32 ms), o que dá 153 frames por janela de 5 s.
2. **Escala mel** com 40 bandas entre 25 e 400 Hz: agrupa as frequências e dá mais resolução às baixas, reduzindo a dimensão.
3. **Logaritmo:** comprime a gama dinâmica, porque os sopros têm muito menos energia que S1/S2 e ficariam "esmagados" numa escala linear.
4. **Z-score por janela** nos dois eixos (tempo e frequência), para remover a escala arbitrária do log-mel.

O resultado é uma "imagem" temp x frequência com 1 canal, que a CNN 2D processa como uma imagem em tons de cinzento.

### 5.2 MFCC + deltas (RNN)

- Parte do mesmo log-mel, mas com um passo de frame maior (64 amostras), o que origina sequências mais curtas (77 passos em vez de 153). As GRU/LSTM treinam melhor e mais depressa com sequências curtas.
- Aplica-se a DCT para obter **13 coeficientes MFCC**, que decorrelaciona as bandas e comprime a informação.
- Acrescentam-se os **deltas** (1.ª derivada no tempo) e **delta-deltas** (2.ª derivada), que descrevem como o espectro varia: 13 x 3 = **39 features** por passo.
- Z-score ao longo do tempo, para cada feauture.
- Cada frame é um passo temporal da sequência.

## 6. Ficheiros Gerados

A seccção 4 do notebook grava em `cache/`:

| Ficheiro | Conteúdo |
|---|---|
| `train.npz`, `val.npz`, `test.npz` | arrays `cnn2d`, `rnn` e `y` (etiqueta do paciente, por janela) |
| `train_info.csv`, `val_info.csv`, `test_info.csv` | uma linha por janela: `pid`, `loc`, `rec`, `y` |

Os `*_info.csv` são necessários para **agregar as previsões por janela numa previsão por paciente**, que é o que a métrica exige. Os ficheiros `.wav` nunca são alterados; a cache é só um resultado derivado e pode ser apagada e regenerada a qualquer momento.

Para carregar nos notebooks dos modelos

```python
import numpy as np, pandas as pd
d = np.load("cache/train.npz")
info = pd.read_csv("cache/train_info.csv", dtype={"pid": str})
X, y = d["cnn2d"], d["y"]     # CNN
# X = d["rnn"]                # RNN
```

## 7. Como reproduzir

1. Descarregar o CirCor 1.0.3 de <https://physionet.org/content/circor-heart-sound/1.0.3/> e colocar em `data/circor/` (deve ficar `data/circor/training_data.csv` e `data/circor/training_data/*.wav`). Os dados não estão no repositório.
2. Instalar as dependências: `pip install tensorflow numpy pandas scipy scikit-learn matplotlib` ou através de um ambiente conda.
3. Abrir `preprocessamento.ipynb` a partir de `data_analysis/` e correr as células por ordem.
4. **Se for alterado algum parâmetro do `Config`**, apagar a pasta `cache/` e voltar a correr a secção 4; caso contrário os modelos treinam com as features antigas sem aviso.

## 8. Limitações

- Rótulos ao nivel do paciente atribuídos a todas as janelas (origina ruído nos rótulos)
- Banda 25-400 Hz corre o risco de cortar sopros de frequências mais elevadas, que enquanto pouco comuns podem existir.
- O z-score elimina a intensidade absolute e é sensível a artefactos de grande amplitude.
- Gravações muito curtas são completadas com zeros gerando silêncio artificial.