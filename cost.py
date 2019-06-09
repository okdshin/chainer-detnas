import chainer
import chainer.links as L
import numpy as np

from chainer_computational_cost import ComputationalCostHook

import detnas

def main():
    net = detnas.DetNASSmallCOCO(n_class=1000)
    #x = np.random.random((1, 3, 1088, 800)).astype(np.float32)
    x = np.random.random((1, 3, 224, 224)).astype(np.float32)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        with ComputationalCostHook(fma_1flop=True) as cch:
            y = net(x)
            cch.show_report(unit='M', mode='table')

if __name__ == '__main__':
    main()
