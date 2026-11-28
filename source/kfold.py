from source.config import *


def split_fold(y, pid, k, fold=0):

    #Outer fold: Data splitter. Object will split fed data into K sets, grouped by patient id
    outer_split = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=0
    )

    inner_split=StratifiedGroupKFold(
        n_splits=6, 
        shuffle=True, 
        random_state=1
    )

    dummy_X = np.zeros(len(y)) #Quantas samples (casos registados) existem.

    #Gera K folds ((train_idx, test_idx)), que são tuplos de np.Array, com dados de features(X), target class(y) e código do paciente(pid)
    #Split apenas das posições (indexes) do dummy array, 
    #isslice avança por todas as folds, ao invés de retornar apenas a fold=0
    train_idx, test_idx = next(islice(outer_split.split(dummy_X, y, pid), fold, None))


    #Garantir que não há data leakage de identidades: um paciente está presente na validação e no treino.
    assert not set(pid[train_idx]) & set(pid[test_idx]), "Data leakage: doentes repetidos entre treino e fold"

    #Inner splitter object

    #Split indexes of only training array
    dummy_X=np.zeros(len(train_idx))

    #Posições dentro do array de treino (train_idx)
    #inner_split.split --> objeto Iterator gera folds step by step. 
    # next() --> Guarda a primeira iteração do Iterator
    #Evita construir "n_splits" e guardar em memória
    #Gera posições relativas dentro de train_idx para o set de treino

    fit_pos, val_pos = next(inner_split.split(dummy_X, y[train_idx], pid[train_idx]))

    #Recolher índices dos registos (linhas do array) para 2 efeitos:
        #FIT: rows used for weigth training
        #IN: rows used for val_wa/epoch
    fit_idx, in_idx = train_idx[fit_pos], train_idx[val_pos]

    return fit_idx, in_idx, test_idx