from math import pi

def kuka_lbr_iiwa_7():
    """Get KUKA LBR iiwa 7 MDH model. (https://github.com/nnadeau/pybotics)"""
    dh = [
        {'alpha':       0, 'delta': 0, 'd': 0, 'a': 0.340},
        {'alpha': -pi / 2, 'delta': 0, 'd': 0, 'a': 0.000},
        {'alpha':  pi / 2, 'delta': 0, 'd': 0, 'a': 0.400},
        {'alpha':  pi / 2, 'delta': 0, 'd': 0, 'a': 0.000},
        {'alpha': -pi / 2, 'delta': 0, 'd': 0, 'a': 0.400},
        {'alpha': -pi / 2, 'delta': 0, 'd': 0, 'a': 0.000},
        {'alpha':  pi / 2, 'delta': 0, 'd': 0, 'a': 0.126},
    ]
    return dh

# ...
