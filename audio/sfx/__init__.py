import scipy
import torch


def preemphasis_np(x, preemphasis_factor=0.9):
    return scipy.signal.lfilter([1, -preemphasis_factor], [1], x)


def inv_preemphasis_np(x, preemphasis_factor=0.9):
    return scipy.signal.lfilter([1], [1, -preemphasis_factor], x)


def mix_with_snr(a, b, snr=20):
    """
    PyTorch Function
    :param a: A Signal
    :param b: B Signal
    :param snr: The ideal A / B Power Ratio, SNR in the case where B is noise
    :return: mixed signals, assuming orthogonal inputs, with same energy amount as A.
    """
    A = torch.sum(a ** 2)
    B = torch.sum(b ** 2)
    c = (10 ** (snr / 5) * A + B) ** 0.5
    d = 10 ** (snr / 10)
    e = A ** 0.5
    s = d * e / c
    t = e / c
    x = a * s + b * t
    return x
