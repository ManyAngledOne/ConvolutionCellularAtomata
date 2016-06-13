import numpy as np
from scipy.signal import fftconvolve

class CARule():
    def __init__(self, kernel, alivecond, deadcond):
        # kernel: convolution kernel
        # alivecond: condition on the convolution which determines if a living cell remains alive
        # deadcond: condition on the convolution which determines if dead cell comes to life
        self.kernel = kernel
        self.cond = np.vectorize(lambda conv,state: (state and alivecond(conv)) or (not state and deadcond(conv)), otypes=[np.int])

class CAState():
    def __init__(self, init_state, rule):
        self.state = init_state
        self.rule = rule

    def evolve(self):
        yield self.state
        while True:
            self.state = self.rule.cond(fftconvolve(self.state, self.rule.kernel, mode='same'), self.state)
            yield self.state

conway = CARule([[1,1,1],[1,0,1],[1,1,1]], lambda x: 1.6<x<3.4, lambda x: 2.6<x<3.4)
init = np.random.choice((0,1), (128,128))
s = CAState(init, conway)
gen = s.evolve()

def foo(n):
    for i in range(n):
        gen.__next__()
