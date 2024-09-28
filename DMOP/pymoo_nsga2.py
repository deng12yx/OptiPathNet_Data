from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import time
from pymoo.core.problem import Problem

import pymoo.gradient.toolbox as anp
import numpy as np


class MyCustomProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=2,  # 变量的数量
            n_obj=2,  # 目标函数的数量
            elementwise_evaluation=True  # 是否逐个评估目标函数值
        )

        self.xl = np.array([0, 0])
        self.xu = np.array([1, 1])
        self.num = 2

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.T
        f1 = x[0]
        f2 = x[1] + x[1]
        out["F"] = anp.column_stack([f1, f2])


def custom_evaluate(x, out, *args, **kwargs):
    f1 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2  # 修改 f1 的计算方式
    f2 = np.sin(10 * np.pi * x[0]) * np.cos(10 * np.pi * x[1])  # 修改 f2 的计算方式

    out["F"] = anp.column_stack([f1, f2])


problem = MyCustomProblem()
problem.n_var = 3
problem.xl = np.array([0, 0, 0])
problem.xu = np.array([1, 1, 1])
# problem = get_problem("ZDT1")
algorithm = NSGA2(pop_size=23, elimate_duplicates=False)
start = time.time()
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)
print(f"Best solution found: \nX = {res.X}\nF = {res.F}")
end = time.time()
# plt.scatter(res.F[:, 0], res.F[:, 1], marker="o", s=10)
# plt.grid(True)
# plt.show()
print('耗时：', end - start, '秒')
