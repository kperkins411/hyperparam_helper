import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from sequence_generators import CosignVals,LinearDecrease,LinearIncreaseVals,ReverseTriangularVals,TriangularVals
else:
    from .sequence_generators import CosignVals,LinearDecrease,LinearIncreaseVals,ReverseTriangularVals,TriangularVals




def get1Cycle_LR_and_Momentum(num_batches, numb_annihlation_batches, annihilation_divisor, max_lr, min_lr, max_momentum, min_momentum):
    '''
    generates LRs as outlined in
    A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS: PART 1 LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY
    :param num_batches: how many discrete learning rates in the triangular phase
    :param numb_LR_annihlation_steps: how many discrete learning rates in the annihlation phase
    :param annihilation_divisor: what the minimum learning rate should wind up as
    :param max_lr: as determined by learning rate finder
    :param min_lr: "
    :param max_momentum:
    :param min_momentum:
    :return: zipped tuple of Learning rates and momentums
    '''

    #first get the learning rates
    lrs = get1CycleVals(num_batches, numb_annihlation_batches, annihilation_divisor, max_lr, min_lr)

    #then lets get the momentums, the anilation steps should have constant momentum, so annihilation_divisor = 1
    NO_ANNIHLATION = 1
    momentums = get1CycleVals(num_batches, numb_annihlation_batches, annihilation_divisor=NO_ANNIHLATION,
                              max_val = max_momentum, min_val =min_momentum,
                              annihlation_val = max_momentum, seq_generator = ReverseTriangularVals())

    return lrs,momentums

def get1CycleVals(numb_batches, numb_annihlation_batches, annihilation_divisor, max_val, min_val, annihlation_val = None,
                  seq_generator = None):
    '''
    :param seq_generator: the type of sequence to generate, defaults to TriangularVals for triangular learning rates
                        can also be ReverseTriangularVals for modulating the momentum
    :param annihlation_val where annihilation process begins,
    :return: list of vals
    '''
    if annihlation_val is None:
        annihlation_val = min_val   #default assumes LRs generated

    if seq_generator is None:
        seq_generator = TriangularVals() #default assumes LRs generated
    vals = []
    vals += seq_generator.getVals(numb_batches, max_val=max_val, min_val=min_val)
    l1 = LinearDecrease()
    vals += l1.getVals(numb_annihlation_batches, max_val=annihlation_val, min_val=annihlation_val / annihilation_divisor)
    return vals



def getCosignAnnealedLinearDecreasingLRs(numb_iterations, numb_steps_per_iteration, max_val, min_val ):
    '''
    demos annealed learning rates using a cosign annealer and linear decreasing learning rate
    can use linear decreasing annealer or cosignLr as well
    :param numb_iterations : how many cycles
    :param numb_steps_per_iteration: how many discrete learning rates per iteration
    :param max_val:
    :param min_val:
    :return: list of learning rates
    '''
    lrs = []
    annealer = CosignVals()
    max_vals = annealer.getVals(numb_iterations=numb_iterations, max_val=max_val, min_val=min_val)
    for max_val in max_vals:
        lr = LinearDecrease()
        lrs += lr.getVals(numb_iterations=numb_steps_per_iteration, max_val=max_val, min_val=min_val)
    return lrs

import matplotlib.pyplot as plt
if __name__ == '__main__':
    vals = getCosignAnnealedLinearDecreasingLRs(numb_iterations=200, numb_steps_per_iteration=20, max_val=1, min_val=0.1)
    vals1 = get1CycleVals(numb_batches=200, numb_annihlation_batches=20, annihilation_divisor=100, max_val=1, min_val=0.1)

    rt = ReverseTriangularVals()
    vals2 = rt.getVals(numb_iterations = 9, max_val=1, min_val=.1)

    plt.xlabel("sample")
    plt.ylabel("learning rate")
    # plt.scatter(range(len(vals)), vals)
    # plt.scatter(range(len(vals1)), vals1)
    #
    # plt.scatter(range(len(vals2)), vals2)
    # plt.show()

    lrs,moms = get1Cycle_LR_and_Momentum(num_batches=200, numb_annihlation_batches=20, annihilation_divisor=100, max_lr=1,
                                         min_lr=0.1, max_momentum=.99, min_momentum=.7)
    plt.scatter(range(len(lrs)), lrs)
    plt.scatter(range(len(moms)), moms)
    plt.show()
