import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from sequence_generators import CosignVals, LinearDecrease, LinearIncreaseVals, ReverseTriangularVals, TriangularVals
    from learning_rate_generators import get1Cycle_LR_and_Momentum
else:
    from .sequence_generators import CosignVals, LinearDecrease, LinearIncreaseVals, ReverseTriangularVals, \
        TriangularVals
    from .learning_rate_generators import get1Cycle_LR_and_Momentum

'''
Implementation of 'Cyclical Learning Rates for Training Neural Networks' by Leslie N. Smith

A scheduler that provides a variable cyclic LR generator.  The cycles can be triangular or Cosign based.
Also implements Max_lr annealing, either linear or cosign based

Example usage given at the bottom of this page

'''
def getNumbImagesInDataLoader(dataloader):
    '''
    how many images are in the pytorch dataloader
    :param dataloader:
    :return:
    '''
    return len(dataloader.dataset)
class Cyclic_Scheduler(object):
    def __init__(self, optimizer,min_lr, max_lr, batch_size = 64, writer =None):
        '''
        :param optimizer:  optimizer, used primarily for applying learning rate to params
        :param min_lr:
        :param max_lr:
        :param batch_size:
        :param writer: tensorboard writer
        '''
        self.optimizer = optimizer   # optimizer layers to which learning rates are applied
        self.min_lr = min_lr
        self.max_lr = max_lr;
        self.batch_size = batch_size
        self.currentLR = min_lr
    def _get_Vals(self):
        raise NotImplementedError
    def batch_step(self):
        raise NotImplementedError
    def get_currentLR(self):
        return self.currentLR


class OneCycle_Scheduler(Cyclic_Scheduler):

    def __init__(self, optimizer,*,min_lr, max_lr, num_batches, numb_annihlation_batches, annihilation_divisor,
                 max_momentum, min_momentum,batch_size = 64, writer =None ):
        super().__init__(optimizer, min_lr, max_lr, batch_size, writer)

        #get all that we need
        self.lrs, self.moms= get1Cycle_LR_and_Momentum(num_batches, numb_annihlation_batches, annihilation_divisor,
                                                       max_lr, min_lr, max_momentum, min_momentum)
        #generator
        self.getVals = self._get_Vals()

    def _get_Vals(self):
        for lr, mom in zip(self.lrs, self.moms):
            yield lr, mom

    def batch_step(self):
        lr,mom = next(self.getVals)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom


class LearningRateFinder(Cyclic_Scheduler):
    '''
    generates a list of linearly increasing learning rates
    use it to find the max and min Learning rates
    '''
    def __init__(self, optimizer,*,min_lr, max_lr, num_batches,  writer =None ):
        super().__init__(optimizer, min_lr, max_lr,  writer)

        #now lets change self.lrs to be just Linear_increase
        li = LinearIncreaseVals()
        self.lrs = li.getVals( numb_iterations = num_batches, max_val = max_lr, min_val = min_lr)

        # generator
        self.getVals = self._get_Vals()

    def _get_Vals(self):
        for lr in self.lrs:
            yield lr
    def batch_step(self):
        self.currentLR = next(self.getVals)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.currentLR

class CyclicLR_Scheduler(Cyclic_Scheduler):
    '''
       Part of stochastic gradient descent with warm restarts
       be careful with the learning rate, set it too high and you blow your model
       Best to use a learning rate finder to get the proper range.

       be sure the number of batches you run is equal to sum(step_size)*2 to get the learning rate to min_lr
       for example, step_size = [5,5,5,5,10,10,10]  then number batches = 50*2=100

       best to use some sort of learning rate finder
    '''
    NUMBER_STEPS_PER_CYCLE = 2

    def __init__(self, optimizer,*,min_lr, max_lr,numb_images_in_dataset, LR,LR_anneal=None,
                 batch_size = 64,step_size=[2], writer =None):
        '''
        :param optimizer:  optimizer, used primarily for applying learning rate to params
        :param min_lr:
        :param max_lr:
        :param numb_images_in_dataset:
        :param LR:  calculates a single cycle of the learning rate
        :param LR_anneal:  anneals the learning rate if present
        :param base_lr:  the minimum learning rate
        :param max_lr:  the maximum learning rate
        :param batch_size:
        :param numb_images:
        :param step_size:number of training images per 1/2 cycle, authors suggest setting it between 2 and 10
                        can be a list, if 4 the whole cycle has 2*(4*numb_images//batch_size) = #iterations/cycle
                        this should sum to total number of epochs or your last epoch has lr somewhere
                        between max_lr and base_lr
        :param writer: tensorboard writer
        usage:
        >>>lr = LinearDecrease()
        >>>anneal = LinearDecrease() # linear annealing
        >>>clr_schedule = CyclicLR_Scheduler(dummyoptimizer, min_lr=.1, max_lr=1, numb_images_in_dataset=100, LR=lr,
                                      LR_anneal=anneal, batch_size=10, step_size=[1, 1, 1, 1])

        >>>vals = list(clr_schedule._get_Vals()) #gets all LRs
        >>>val = clr_schedule.batch_step()     #applies single learning rate to optimizer param_groups
        '''
        super().__init__(optimizer,min_lr, max_lr,batch_size, writer)

        #learning rate sequence generator
        self.LR = LR

        #learning rate annealer sequence generator
        self.LR_anneal = LR_anneal

        #how many steps
        self.step_size = step_size
        self.numb_images = numb_images_in_dataset

    def _get_Vals(self):
        numb_batches_per_epoch = self.numb_images//self.batch_size

        # how many annealing values between max_lr and min_lr
        max_lrs = [self.max_lr]*len(self.step_size)
        if self.LR_anneal is not None:
            max_lrs = self.LR_anneal.getVals(len(self.step_size), max_val=self.max_lr, min_val=self.min_lr)

        for max_lr, step in zip(max_lrs, self.step_size):
            # how many batches per epoch
            numb_epochs = CyclicLR_Scheduler.NUMBER_STEPS_PER_CYCLE *step
            numb_batches = numb_epochs*numb_batches_per_epoch

            #get some learning rates
            lrs = self.LR.getVals(numb_batches,max_val=max_lr, min_val = self.min_lr )
            for lr in lrs:
                yield lr

    def batch_step(self):
         for param_group, lr in zip(self.optimizer.param_groups, self._get_Vals()):
            param_group['lr'] = lr
         self.cur_lr = lr   #used in learning rate finder

import matplotlib.pyplot as plt


if __name__ == '__main__':
    dummyoptimizer = 3  #bogus for class

    # lr = LinearDecrease()

    #--- test without annealing
    # clr_schedule = CyclicLR_Scheduler(dummyoptimizer,min_lr=.1, max_lr=1,numb_images_in_dataset=100, LR=lr,
    #              batch_size = 10,step_size=[1,1,1,1])

    #---test with cosign annealing
    # anneal = CosignVals()     #cosign annealing
    # anneal = LinearDecrease() # linear annealing
    # clr_schedule = CyclicLR_Scheduler(dummyoptimizer, min_lr=.1, max_lr=1, numb_images_in_dataset=1000, LR=lr,
    #                                   LR_anneal=anneal, batch_size=10, step_size=[1]*20)
    # plt.xlabel("sample")
    # plt.ylabel("learning rate")
    # plt.scatter(range(len(vals)), vals)
    # plt.show()

    #--- test with onecycle
    clr_schedule = OneCycle_Scheduler(dummyoptimizer, num_batches=200, numb_annihlation_batches=20, annihilation_divisor=100, max_lr=1,
                                         min_lr=0.1, max_momentum=.99, min_momentum=.7)
    vals = list(clr_schedule._get_Vals())
    lrs, moms = zip(*vals)

    plt.xlabel("sample")
    plt.ylabel("learning rate")
    plt.scatter(range(len(lrs)), lrs)
    plt.scatter(range(len(moms)), moms)

    plt.show()

