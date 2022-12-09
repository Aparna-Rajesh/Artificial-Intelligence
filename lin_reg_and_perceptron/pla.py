import pandas as pd
import numpy as np
import csv
import sys


class Percep:
    def __init__(self, write_to, formatted_data):
        self.calc_weights = [0] * len(formatted_data[0])
        self.signs = formatted_data[:, -1]  # only want the true labels
        self.write_to = write_to
        self.data_with_ones = np.insert(formatted_data, 2, 1, axis=1)

    def do_the_thing(self):
        with open(self.write_to, 'w') as w:
            writer = csv.writer(w, delimiter=',')
            counter, has_converged = 0, False

            '''
            Convergence happen when the weights do not change anymore

            NOTE: adding a half a million iteration convergence max in case we get some 
            input that takes horribly long or is actually unable to converge...
            '''
            while not has_converged and counter <= 500000:
                as_ints = []
                for x in self.calc_weights:
                    as_ints.append(int(x))
                writer.writerow(as_ints)

                has_converged = True
                index = 0
                for ith_row in self.data_with_ones:
                    feature, y_i = ith_row[0:len(
                        ith_row)-1], self.signs[index]
                    prediction = np.dot(self.calc_weights, feature)

                    # if pred, f(xi) does not look like the true label; it would be 1 if they matched
                    if y_i * prediction <= 0:
                        print("miss!", ith_row)
                        '''
                        Mistake on positive: add x to weight vector.
                        Mistake on negative: substract x from weight vector.
                        '''
                        #print("before", self.calc_weights, y_i, feature)
                        new_weights = []
                        for w in range(len(self.calc_weights)):
                            # yi serves as the switch
                            new_weights.append(
                                self.calc_weights[w] + feature[w] * y_i)
                        self.calc_weights = new_weights
                        # print(self.calc_weights)
                        # we had found a mismatch, so this iteration is definitely not convergence
                        has_converged = False
                    index += 1
                counter += 1
            # we will leave the last one as a decimal values just to be more precise
            writer.writerow(self.calc_weights)
        return has_converged


def main():
    '''YOUR CODE GOES HERE'''
    if len(sys.argv) != 3:
        print("Not enough arguments.")
        exit(1)

    p = Percep(sys.argv[2], np.genfromtxt(
        sys.argv[1], delimiter=',', dtype=float))
    p.do_the_thing()


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
