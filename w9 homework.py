from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        state = self.start_state
        output = []
        for inp in input_seq:
            state = self.transition_fn(state, inp)
            output.append(self.output_fn(state))
        return output  
        


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = None # Change

    def transition_fn(self, s, x):
        # Your code here
        return ((s[1]+sum(x))%2, (s[1]+sum(x))//2)

    def output_fn(self, s):
        # Your code here
        return s[0]


class Reverser(SM):
    start_state = None
    end_index = -1
    index = 0
    sequence1 = []
    seq1_index = 0

    def transition_fn(self, s, x):
        # Your code here
        self.sequence1.append(x)
        self.end_index += 1
        if x == 'end':
            self.index = self.sequence1.index('end')
        if self.index ==0:
            return None
        elif (self.index >0) and (self.end_index >= self.index) and (self.end_index < self.index*2):
            self.seq1_index +=1
            return self.sequence1[self.index-self.seq1_index]
        else:
            return None

    def output_fn(self, s):
        # Your code here
        return s


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0 = Wo_0
        self.l = self.Wsx.shape[1]
        self.m = self.Wss.shape[1]
        self.n = self.Wo.shape[1]
        self.start_state = np.zeros((self.n, 1))
        self.f1 =f1
        self.f2 = f2
        
    def transition_fn(self, s, i):
        # Your code here
        return self.f1(self.Wss@s+self.Wsx@i+self.Wss_0)

    def output_fn(self, s):
        # Your code here
        return self.f2(self.Wo@s+self.Wo_0)
