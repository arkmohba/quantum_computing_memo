# pyquboを使う
from pyqubo import Array, Constraint
import numpy as np
from scipy.stats import poisson, norm
from qubo_solver import BaseSolver, OjijCpuSolver


def create_qubo(p_ij, f_ii, n_item, omega=1, M=10):
    # PyQUBOで定式化
    x = Array.create('x', shape=(n_item, n_item), vartype='BINARY')
    # 第一項の計算
    H_p = 0
    for i in range(n_item):
        H_p -= x[i].dot(p_ij[i])

    # 第二項の計算
    H_fd = 0
    for i1 in range(0, n_item-1):
        for i2 in range(i1+1, n_item):
            val = f_ii[i1][i2]
            for j1 in range(n_item):
                for j2 in [j1-1, j1+1]:  # d_jjに対応
                    if j2 < 0 or j2 >= n_item:
                        # j2が範囲外ならスキップ
                        continue
                    H_fd += val * x[i1][j1] * x[i2][j2]
    # 制約条件（第三項、第四項）の作成
    H_const = 0
    one_vec = np.ones(n_item)
    # ある順番に表示するアイテムは1個の制約
    for order_ in range(n_item):
        H_const += Constraint((x[:, order_].dot(one_vec) - 1)
                              ** 2, label='sum-equal')

    # アイテムは1回だけ表示の制約
    for item_ in range(n_item):
        H_const += Constraint((x[item_].dot(one_vec) - 1)
                              ** 2, label='sum-equal')

    return (H_p + omega * H_fd + M*H_const).compile().to_qubo()


def decode_sample(result_sample, item_size):
    # 行列にデコード
    result_matrix = np.zeros((item_size, item_size), dtype=int)
    for i in range(item_size):
        for j in range(item_size):
            # PyQUBOを使うと変数名でインデックスされている。
            result_matrix[i][j] = result_sample['x[{}][{}]'.format(i, j)]
    return result_matrix


def validate_mat(mat: np.array):
    # 行方向に1の個数をカウント
    low_count = np.count_nonzero(mat == 1, axis=0)
    success = np.all(low_count == 1)
    if not success:
        # print("Error: being different places in the same time")
        # print(mat)
        return False

    # 列方向に1の個数をカウント
    col_count = np.count_nonzero(mat == 1, axis=1)
    success = np.all(col_count == 1)
    if not success:
        # print("Error: revisit same place")
        # print(mat)
        return False
    return True

# ガウシアン版
# def get_pij(n_item: int):
#     p_ij = np.zeros((n_item, n_item))
#     xs = np.arange(n_item)
#     for i in range(n_item):
#         p_ij[i] = norm.pdf(xs, scale=0.5, loc=i)
#         p_ij[i] /= p_ij[i].max()
#     return p_ij


def get_pij(n_item: int):
    # ポワソン分布版
    p_ij = np.zeros((n_item, n_item))
    xs = np.arange(n_item + 1)
    for i in range(n_item):
        p_ij[i] = poisson.pmf(xs, i+1)[1:]
        p_ij[i] /= p_ij[i].max()
    return p_ij


def get_optimized_order(similarity_matrix: np.ndarray, solver: BaseSolver, omega=1, M=10):
    """similarity_matrixをもとに似たベクトルは並ばないようにする。

    Args:
        similarity_matrix (np.ndarray): [description]
        solver (BaseSolver): [description]
        omega (int, optional): [description]. Defaults to 1.
        M (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_item = len(similarity_matrix)
    p_ij = get_pij(n_item)
    p_ij /= p_ij.max()
    f_ii = similarity_matrix
    f_ii /= f_ii.max()

    qubo = create_qubo(p_ij, f_ii, n_item, omega, M)
    x_shape = (n_item, n_item)
    res = solver.solve(q=qubo[0], x_shape=x_shape)
    success = validate_mat(res)
    orders = res.argmax(axis=1)
    return success, orders


class OrderSolver:
    def __init__(self, num_reads=100):
        self.solver = OjijCpuSolver(
            num_reads=num_reads)

    def rebalance_order(self, similarity_matrix, target_list, omega=1, M=100):
        """順番を並び替える

        Args:
            similarity_matrix (np.ndarray): 候補アイテム間の距離行列。
            target_list (list): 並び替えたい対象のリスト。

        Returns:
            list: 並び替えた後のリスト
        """
        success, new_order = get_optimized_order(
            similarity_matrix, self.solver, omega=omega, M=M)
        output_list = [target_list[i] for i in new_order]
        return success, output_list
