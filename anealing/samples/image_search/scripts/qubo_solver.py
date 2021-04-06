'''
x[i][j]形式のqubitに対するquboのソルバー
pyquboベースとする
'''
import openjij as oj
import numpy as np


class BaseSolver:
    def __init__(self, mode="qubo"):
        self.mode = mode

    def solve_qubo(q: dict, x_shape=[]):
        """pyquboから生成された{x[i][j]:val}形式の辞書を引数にquboを解く関数

        Args:
            q (dict): [description]
        """
        print("qubo sovler not implemented!")
        pass

    def solve_ising(h, j, x_shape=[]):
        # 今回こちらは使わないが、一応作成
        print("Ising sovler not implemented!")
        pass

    def solve(self, q=None, h=None, j=None, x_shape=[]):
        if self.mode == "qubo":
            if q is None:
                raise ValueError("q should be set if solving qubo problem.")
            return self.solve_qubo(q, x_shape=x_shape)
        else:
            if h is None and j is None:
                # メモ：and ではなくorにすべき？
                raise ValueError(
                    "h or j should be set if solving ising problem.")
            return self.solve_ising(h, j, x_shape=x_shape)


def decode_pyqubo_sample(result_sample, x_shape, prefix="x"):
    # 行列にデコード
    result_matrix = np.zeros(x_shape, dtype=int)
    format_str = prefix + '[{}][{}]'
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            # PyQUBOを使うと変数名でインデックスされている。
            result_matrix[i][j] = result_sample[format_str.format(i, j)]
    return result_matrix


class OjijCpuSolver(BaseSolver):
    def __init__(self, mode="qubo", prefix="x", num_reads=1):
        super().__init__(mode)
        self.solver = oj.SASampler()
        self.prefix = prefix
        self.num_reads = num_reads

    def solve_qubo(self, qubo, x_shape):
        sampleset = self.solver.sample_qubo(qubo, num_reads=self.num_reads)

        result = decode_pyqubo_sample(
            sampleset.first.sample, x_shape, prefix=self.prefix)
        return result
