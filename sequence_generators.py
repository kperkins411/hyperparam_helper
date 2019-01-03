import math
import numpy as np

class Vals(object):
    '''
    Base class, returns a single cycle of values
    '''
    def __call__(self, *args, **kwargs):
        return self.getVals()

    def getVals(self, numb_iterations, max_val, min_val):
        '''
        :param numb_iterations: number total samples
        :param max_val: upper
        :param min_val: lower
        :return: list of learning rates, numb_iterations long
        '''
        raise NotImplementedError

class CosignVals(Vals):
    def getVals(self, numb_iterations, max_val, min_val):
        '''
        Cosign that starts at max_val and decreases to min_val
        '''
        # then translate so ranges between low_lr and high_lr
        data = ( (np.cos(np.linspace(0, np.pi, numb_iterations)))+1) *(max_val - min_val)/2 + min_val
        return data

class TriangularVals(Vals):
    def getVals(self, numb_iterations, max_val, min_val):
        #if odd numb_iterations add extra to first half
        extra = numb_iterations%2

        # determines the halfway point
        step_size = numb_iterations // 2

        first_half = np.linspace(min_val, max_val, (step_size+extra))
        second_half = np.linspace(min_val, first_half[step_size + extra - 2], step_size)  # note range to second from end

        return np.concatenate((first_half, np.flip(second_half))).tolist()

class ReverseTriangularVals(Vals):
    def getVals(self, numb_iterations, max_val, min_val):
        # if odd numb_iterations add extra to first half
        extra = numb_iterations % 2

        # determines the halfway point
        step_size = numb_iterations // 2

        first_half = np.flip(np.linspace(min_val, max_val, (step_size+extra)))
        second_half = np.linspace(first_half[step_size + extra - 2], max_val, step_size)  # note range to second from end
        return np.concatenate((first_half, second_half)).tolist()

#useful for learning rate finder
class LinearIncreaseVals(Vals):
    def getVals(self, numb_iterations, max_val, min_val):
        return np.linspace(min_val, max_val, numb_iterations).tolist()

class LinearDecrease(LinearIncreaseVals):
    def getVals(self, numb_iterations, max_val, min_val):
        return np.flip(super().getVals(numb_iterations, max_val, min_val)).tolist()
