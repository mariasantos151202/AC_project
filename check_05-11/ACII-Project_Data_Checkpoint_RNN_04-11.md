
creation date:  2026-11-04 16:43
last modification date: Wednesday 4th Nov 16:43

## Summary: Data analysis (João, 04-11)

- Heart sounds: dados sequenciais e periódicos;
- Duração das gravações (variable length)
	- Analisar sequências de duração distinta: padding, packing;
		- Window slicing para aumentar eficiência de processamento em batch. Separar gravações em gravações curtas de 5-10s;  Fixed-length feature vector;
			- Classificação destas gravações
			- Agrupamento de classificações de gravações;
	- Sequence length:
		- Raw audio: `4 kHz × 30 s = 120,000 samples`
- LSTM: treino lento, com elevado risco de overfitting se o dataset for pequeno. Deve ser combinado com features convolucionais (CNN+ LSTM)
- CNN: treino rápido, adaptado para classificação de áudio; transfer learning;

## Dataset analysis

- `df.shape()=(93,23)`;
- Não há registos duplicados de pacientes;
- `Absent:50 Present:33 Unknown:9`
- `df["n_rec_locais"] = df["Recording locations:"].str.split("+").str.len()`
	- `Recording locations`: locais anatómicos onde foram feitas auscultações; 
	- `df["n_rec_locais"]`: contagem do número de locais  onde foram realizadas auscultações/patient id;
	- Contagem do número de registos por contagem do número de locais:

## Dataset processing for RNN

Domain-specific feature selection: seleção de variáveis com elevada correlação com outcome de previsão. 

Usar um layer de autoencoder ou PCA para comprimir as features num espaço latente (diminuir dimensionalidade);

RNN exige reorganização dos arrays de input em 3D Tensor

## Summary: RNN (LSTM)


## Limitations of RNN : challenge analysis

Maioria das soluções propostas não utiliza LSMT. CNN foram amplamente explorado entre os 10 primeiros classificados. 

---

# Presented papers in Computing in Cardiology

Link: https://physionet.org/files/challenge-2022/1.0.0/papers/index.html

## A CNN-BiLSTM model with attention mechanism for earthquake prediction

Link: https://link.springer.com/article/10.1007/s11227-023-05369-y

## Detection of Heart Murmurs in Phonocardiograms with Parallel Hidden Semi-Markov Models

Link: https://ieeexplore.ieee.org/document/10081712; [[McDonald_Heart_Murmur.pdf]]

- Segmentation: fixed-length feature vector
- Bidirectional GRU: previsão dos segmentos modificados (modified segmentation labels) das features extraídas. Estratégia Greedy de previsão; Noise aumenta falsos positivos na tarefa de classificação de sopro cardíaco.
- Parallel Hidden Semi-Markov: reduzir falsos positivos do RNN 

---

