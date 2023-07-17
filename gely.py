import numpy as np
from functools import reduce


def _is_frequent(pattern, D, threshold):
    c = 0
    if len(pattern) == 0:
        return False, c
    for d in D:
        if pattern.issubset(d):
            c += 1
    is_frequent = c >= threshold
    return is_frequent, c


def _it(X, D) -> set:
    '''return: all transactions that contain all items in X'''

    tids = []

    for x in X:  # set of items
        _tids = []
        for tid, y in enumerate(D):
            if x in y:
                _tids.append(tid)
        if len(_tids) > 0:
            tids.append(set(_tids))

    if len(tids) == 0:
        return set()
    _tids_all_x_appear_in = set.intersection(*tids)
    return _tids_all_x_appear_in


def _ti(Y, D) -> set:
    '''return: all items common to all transactions in Y'''
    '''_ti(_it(X)) -> can never be empty (if |Y| > 0), because at least all X have to be returned'''
    if len(Y) == 0:
        return set()
    iids = [D[tid] for tid in Y]
    return set.intersection(*iids)


## frequency check and closure operator using binary matrix instead of sets and list


def _ti_binary_matrix(Y, B) -> set:
    iids = B[list(Y), :]
    intersected = reduce(lambda a,b: np.bitwise_and(a, b), list(iids))
    intersected = np.argwhere(intersected).squeeze()
    if len(intersected.shape) == 0:
        intersected = np.expand_dims(intersected, 0)
    intersected = list(intersected)
    return set(intersected)


def _it_binary_matrix(X, B) -> set:
    tids = [set(list(np.argwhere(B[:, x]).squeeze(1))) for x in X]
    return set.intersection(*tids)


def _is_frequent_binary(itemset, B, threshold):
    if len(itemset) == 0:
        return False, 0
    _itemset_binary = np.zeros(B.shape[1], dtype=int)
    _itemset_binary[list(itemset)] = 1
    match = B @ _itemset_binary
    match = match == len(itemset)
    n_occurences = np.sum(match)
    is_frequent = n_occurences >= threshold
    return is_frequent, n_occurences


def list_closed(C, N, i, t,
                  D, I, T, results, closure=None, D_binary=None):

    if closure is None:
        closure = lambda X: _ti(_it(X, T, D), I, D)

    idx = np.argwhere(I == i).squeeze()  # {k in I\C : k >= i}
    X = I[idx:]
    X = set(X) - C

    if len(X) > 0:
        _i_prime = min(X)
        _C_prime: set = closure(C.union({_i_prime}))

        if D_binary is not None:
            is_frequent, count = _is_frequent_binary(_C_prime, D_binary, t)
        else:
            is_frequent, count = _is_frequent(_C_prime, D, t)

        if is_frequent and len(_C_prime.intersection(N)) == 0:
            # print(_C_prime, count)
            results.append((_C_prime, count))
            idx = np.argwhere(I == _i_prime).squeeze()
            if idx >= len(I)-1:  # we reached item with highest ordinal
                return
            _next_i = I[idx + 1]  # _i_prime + 1  # leads to Index Out Of Range Error on I
            list_closed(_C_prime, N, _next_i,  # What if min(X) == max(I)?
                        t, D, I, T, results, closure)

        idx = np.argwhere(I == _i_prime).squeeze()+1  # {k in I\C: k > _i_prime}
        Y = I[idx:]  # no out of bounds, just empty
        Y = set(Y) - C
        if len(Y) > 0:
            _i_prime_prime = min(Y)
            list_closed(C, N.union({_i_prime}), _i_prime_prime, t,
                        D, I, T, results, closure)

    return


def gely(B, threshold, use_binary=False, remove_copmlete_transactions=True, targets=None, verbose=True):
    '''

    :param D: Is a binary matrix;
              Columns = Items, index used as ordering
              Tows = Transactions, index used as tid
    :param threshold: frequency threshold used to determine if an itemset is considered frequent
    :return:
    '''

    if targets is not None:
        raise NotImplementedError

    n_full_transactions = None
    if remove_copmlete_transactions:
        n_I = B.shape[1]
        _T_sizes = np.sum(B, 1)
        unfull_transactions = _T_sizes < n_I
        D = B[unfull_transactions]
        n_full_transactions = np.sum(-1*(unfull_transactions-1))
        print(f"removed {n_full_transactions} transactions that contained all items")
        threshold = threshold - n_full_transactions
        assert threshold > 0
        print(f"adapted threshold to {threshold+n_full_transactions} - {n_full_transactions} = {threshold}")
    else:
        D = B

    _size_T, _size_I = D.shape
    T, I = np.arange(_size_T), np.arange(_size_I)
    #  remove items that never occur in any transaction in D (ie, filter zero columns)
    _non_zero_cols = np.argwhere(np.sum(D, 0) > 0).squeeze()
    I = I[_non_zero_cols]
    _t = threshold

    # transform binary rows into transactions represented by lists of items (as indices)
    _D = []
    for _i, t in enumerate(D):
        t_iids = np.argwhere(t).squeeze()
        if len(t_iids.shape) == 0:
            t_iids = np.expand_dims(t_iids, 0)
        _D.append(list(t_iids))
    _D = [set(d) for d in _D]

    if use_binary:
        D = D.astype(int)
        closure = lambda X: _ti_binary_matrix(_it_binary_matrix(X, D), D)  # D not _D !
    else:
        closure = lambda X: _ti(_it(X, _D), _D)

    _fcis = []
    list_closed(
        C=set(), N=set(), i=min(I), t=_t, D=_D, I=I, T=T,
        results=_fcis, closure=closure, D_binary=D if use_binary else None
    )
    # return list sorted by (largest itemsets, larger support)
    _fcis = sorted(_fcis, key=lambda x: (len(x[0]), x[1]), reverse=True)
    if n_full_transactions is not None:
        _fcis = [(f[0], f[1]+n_full_transactions) for f in _fcis]
    return _fcis


def gely_test():
    D = ['abde', 'bce', 'abde', 'abce', 'abcde', 'bcd']
    support_thresh = 3
    I = ['a', 'b', 'c', 'd', 'e']
    _I = {k: v for v, k in enumerate(I)}

    T = np.arange(len(D))
    B = np.zeros((len(T), len(I)))
    for tid, t in zip(T, D):
        for _iid, i in enumerate(I):
            B[tid, _iid] = 1 if i in t else 0

    assert np.all(B ==
                  np.array([
                            [1, 1, 0, 1, 1],
                            [0, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0]
                        ])
                  )

    fcis = gely(B, support_thresh)
    for f, c in fcis:
        print(f"{[I[i] for i in f]} x {c}")
    pass


if __name__ == '__main__':
    gely_test()
