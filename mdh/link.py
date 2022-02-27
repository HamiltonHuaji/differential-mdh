import torch as tf


class RevoluteLink:
    def __init__(self, **kwargs):
        if "a" not in kwargs:
            kwargs["a"] = 0.
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.
        if "d" not in kwargs:
            kwargs["d"] = 0.
        if "delta" not in kwargs:
            kwargs["delta"] = 0.
        if "param" in kwargs:
            self.param = kwargs["param"]
            assert isinstance(kwargs["param"], tf.Tensor)
            assert self.param.shape == (4,)
        else:
            self.param = tf.Tensor((
                float(kwargs["a"]),
                float(kwargs["alpha"]),
                float(kwargs["d"]),
                float(kwargs["delta"])
            ))

    def __repr__(self) -> str:
        a, alpha, d, delta = self.param.detach().numpy()
        return f"RevoluteLink(a={a}, alpha={alpha}, d={d}, delta={delta})"

    def transform(self, theta):
        crx = tf.cos(self.param[1])
        srx = tf.sin(self.param[1])
        crz = tf.cos(theta + self.param[3])
        srz = tf.sin(theta + self.param[3])
        d = self.param[2]
        a = self.param[0]

        transform = tf.stack([
            tf.stack([crz, -srz, tf.tensor(0), a]),
            tf.stack([crx * srz, crx * crz, -srx, -d * srx]),
            tf.stack([srx * srz, crz * srx, crx, d * crx]),
            tf.stack([tf.tensor(0), tf.tensor(0), tf.tensor(0), tf.tensor(1)]),
        ])
        return transform

    def dump(self):
        return {
            "a": self.param[0].item(),
            "alpha": self.param[1].item(),
            "d": self.param[2].item(),
            "delta": self.param[3].item(),
        }

    def load(self, obj):
        self.param.data[0] = tf.tensor(obj["a"], dtype=tf.float64).data
        self.param.data[1] = tf.tensor(obj["alpha"], dtype=tf.float64).data
        self.param.data[2] = tf.tensor(obj["d"], dtype=tf.float64).data
        self.param.data[3] = tf.tensor(obj["delta"], dtype=tf.float64).data

    def __sub__(self, other: "RevoluteLink"):
        return RevoluteLink(param=self.param-other.param)


if __name__ == "__main__":
    l = RevoluteLink()
    print(l)
    l.load({
        'alpha': 0.123,
        'a': 0.6127,
        'delta': 0.456,
        'd': 0.567
    })
    print(l, l.dump())
    params = tf.rand((5, 4))
    q = RevoluteLink(param=params[0])
    print(q, q.dump())
    q.load({
        'alpha': 0.123,
        'a': 0.6127,
        'delta': 0.456,
        'd': 0.567
    })
    print(q, q.dump())
    import numpy as np
    std = np.array([[0.79243837, -0.60995199, 0., 0.6127],
                    [0.60534382, 0.78645153, - 0.12269009, - 0.06956528],
                    [0.07483506, 0.09722434, 0.99244503, 0.56271633],
                    [0., 0., 0., 1.]])
    diff = q.transform(0.2).detach().numpy()-std
    abse = np.sum(np.abs(diff))
    print(f"test passed: {abse<1e-5}")
