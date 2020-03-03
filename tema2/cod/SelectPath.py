import sys
import numpy as np


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    pathway = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if pathway[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif pathway[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = pathway[-1][1] + opt
        pathway.append((line, col))

    return pathway


def select_greedy_path(E):
    col = np.argmin(E[0])
    path = [(0, col)]
    for i in range(1, E.shape[0]):
        if path[-1][1] == 0:
            opt = np.argmin([E[i, 0], E[i, 1]])
        elif path[-1][1] == E.shape[1] - 1:
            opt = - np.argmin([E[i, -1], E[i, -2]])
        else:
            opt = np.argmin([E[i, path[-1][1] - 1], E[i, path[-1][1]], E[i, path[-1][1] + 1]]) - 1
        col = path[-1][1] + opt
        path.append((i, col))

    return path


def select_dynamic_path(E):
    M = np.zeros(E.shape)
    M[0, :] = E[0, :]
    for i in range(1, M.shape[0]):
        M[i, 0] = min(M[i - 1, 0], M[i - 1, 1]) + E[i, 0]
        M[i, -1] = min(M[i - 1, -1], M[i - 1, -2]) + E[i, -1]
        for j in range(1, M.shape[1] - 1):
            M[i, j] = min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1]) + E[i, j]

    path = []
    line = M.shape[0] - 1
    col = np.argmin(M[-1])
    path.append((line, col))
    while line > 0:
        line = line - 1
        if col == 0:
            offset = np.argmin([M[line, col], M[line, col + 1]])
        elif col == M.shape[1] - 1:
            offset = - np.argmin([M[line, -1], M[line, -2]])
        else:
            offset = np.argmin([M[line, col - 1], M[line, col], M[line, col + 1]]) - 1
        col = col + offset
        path.append((line, col))

    path.reverse()

    return path


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)