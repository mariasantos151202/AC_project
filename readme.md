Projeto de AC 2 - cnn

**estrutura da pasta: resultados_cnn**

-  results_cnn_1_split.ipynb: contém conclusões dos reuslatdos que obtivemos nos testes e validações com os modelos sem usar cross-validation. Ainda vou elaborar mais as explicações para vcs entenderem mellhor algumas coisas !

- analise resultados/imp-diff-seeds

    - contém os resultados do improved corrido em diferentes seeds. Porque? Para confirmar que os resulatdos bons feitos com o 1 split eram consistentes em seeds diferentes e que não eram apenas sorte. (smoothed_s... 42, 43, 44)
    - abaltion-diff-seeds.py --> avalia através dos ficheiros obtidos dos smoothed_s.. o impacto que cada alteração teve no resultado: compara o top k com o mean e o weighted com o unweighted.
    - logs das seeds --> tem os resultados das epochs de cada run em cada seed.

- analise resultados --> metrics-1-split-diff-seeds.py
    - retira os valores das diferentes runs e calcula metricas como a media, desvio padrão.

- analise resultados/runs-cv
    - cv/ --> logs_cv.. (0,1,2,3,4) : logs de todos as cross-validation
    - metrics_cv.py --> script que vai calular metricas como media e desvio padrão dos 5-folds.

- analise_resultados/cnn_variations:
    - ficheiros com cada variação da cnn. Falta a resnet

- cnn_structured.py : contém as classes todas que uso para treinar, agregar os dados, modelo, resnet ...

- kfold_cv.py: script para fazer a cross validation com 5 folds. vai devolver os logs naquela pasta das runs-cv

- tests.ipynb: contém as várias variações em forma de jupyter




### **CNN IMPROVED**

Na verdade não alterei nada na arquitetura da CNN2D. Apenas alterei coisas na estratégia de treino.

Quando corri o baseline ( sem as alterações da estratégia de treino ) tivemos um score de wa de cerca de 0.67, com a seguinte matriz de confusão:

RESULTADOS: VALIDAÇÃO (BASELINE) | Estratégia: MEAN

- WA_murmur: 0.6778

                     | True Present |True Unknown | True Absent
        Pred Present |          27  |          4  |        62
        Pred Unknown |           0  |          6  |        13
        Pred Absent  |           0  |          0  |        30

Aqui estamos a agrupar pela média das probabilidades por janela temporal. Isto faz com que as janelas que apresnetam sopro visivek fiquem diluidas nas restantes janelas que não têm sopro.
Além disso, estamos a fazer:
score = pr * METRIC_W if self.decision == "weighted" else pr
            y_pred = score.argmax(axis=1)
neste caso estamos a usar weighted, ou seja, estamos a multiplicar as probabilidades pelo vetor dos pesos [5; 3; 1] antes de passar pelo argmax. Itso faz com  as probailidades que já estavam inflacionadas antes sejam ainda mais esacerbadas por esta multiplicação e com que seja precisa apenas uma probabilidade muito baixa da presente para que o o output seja present ( está explicado no results_cnn_1_split.ipynb)


### O que alterámos do baseline para o Improved?

**No treino**
- **Suavização dos pesos de classe**: em vez do inverso da frequência (N / (3·N_i)), usamos a sua raiz quadrada, normalizada para média 1. Os pesos multiplicam a loss de cada janela, e assim o reforço das classes raras é menor e as probabilidades da rede distorcem menos a prevalência real.

**Na previsão (que não exige retreinar)**
- **Agregação Top-K em vez da média**: por doente, as probabilidades de Present e Unknown são a média das 3 janelas com valores mais altos (top k =3 ) (a de Absent continua a ser a média de todas), e depois renormalizam-se. Evita que um sopro audível em poucas janelas seja diluído pelas restantes.
- **Decisão `unweighted` em vez de `weighted`**: o argmax passa a ser aplicado diretamente às probabilidades do doente, sem as multiplicar por [5, 3, 1]. O WA continua a ser calculado com os pesos 5:3:1 na avaliação.

