import numpy as np
import pandas as pd
import sys

import warnings

warnings.filterwarnings("ignore")

'''
I literally had to watch so many youtube tutorials to figure
out what's going on.

I'm so sorry if my code is mush.
'''


class LinReg:
    def __init__(self, age_w, height, print_to):
        self.age_w = age_w
        self.height = height
        self.result = []
        self.print_to = print_to

    def normalize(self):
        mu = np.mean(self.age_w, axis=0, dtype=np.float64)
        std = np.std(self.age_w, axis=0)
        self.age_w = (self.age_w - mu) / std
        self.age_w = np.column_stack([np.ones(len(self.age_w)), self.age_w])

    def gradient_descent(self, age_weight, height, a):
        max_check = 100
        the_len, wt_calc = len(age_weight), np.zeros(3)
        for x in range(int(max_check)):
            f_of_x = age_weight.dot(wt_calc)
            beta_j = age_weight.T.dot(f_of_x - height) * a / the_len
            wt_calc -= beta_j
        loss = np.sum((age_weight.dot(wt_calc) - height)**2)
        loss = loss / (2*the_len)
        return wt_calc, loss

    def gradient_descent_find(self, age_weight, height, a, total_runs):
        the_len, wt_calc = len(age_weight), np.zeros(3)
        for x in range(int(total_runs)):
            f_of_x = age_weight.dot(wt_calc)
            beta_j = age_weight.T.dot(f_of_x - height) * a / the_len
            wt_calc -= beta_j
        loss = np.sum((age_weight.dot(wt_calc) - height)**2)
        loss = loss / (2*the_len)
        return wt_calc, loss

    def run(self):
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            gottem = self.gradient_descent(self.age_w, self.height, alpha)
            self.result.append(
                [alpha, 100, gottem[0][0], gottem[0][1], gottem[0][2]])

        self.find_best()

    def find_best(self):
        opt_res = None
        opt_loss = float("inf")
        for alpha in np.linspace(0.05, 5, 20):
            for max_iter in np.linspace(10, 1000, 200):
                max_iter = int(max_iter)
                gottem = self.gradient_descent_find(
                    self.age_w, self.height, alpha, max_iter)
                if gottem[1] < opt_loss:
                    opt_res = [alpha, max_iter, gottem[0]
                               [0], gottem[0][1], gottem[0][2]]
                    opt_loss = gottem[1]

        self.result.append(opt_res)
        result_out = pd.DataFrame(self.result)
        result_out.to_csv(self.print_to, header=None, index=None)


def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    if len(sys.argv) != 3:
        print("Not enough arguments.")
        exit(1)

    readfrom, writeto = sys.argv[1], sys.argv[2]

    collected = pd.read_csv(readfrom, header=None, dtype=np.float64)
    data_np = collected.values.astype(np.float64)
    ages_weights, heights = data_np[:, :-1], data_np[:, -1]
    lr = LinReg(ages_weights, heights, writeto)
    lr.normalize()
    lr.run()


if __name__ == "__main__":
    main()
