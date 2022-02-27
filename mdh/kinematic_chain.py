from .link import RevoluteLink
from typing import Dict, List
import torch as tf
import numpy as np

class KinematicChain:
    def __init__(self, joints: List[RevoluteLink]):
        assert len(joints) >= 1
        self.joints = joints

    @classmethod
    def from_tensor(cls, t: tf.Tensor):
        assert t.shape[1] == 4
        joints = [RevoluteLink(param=t[i]) for i in range(t.shape[0])]
        return KinematicChain(joints)

    @classmethod
    def obj_to_tensor(cls, l: List[Dict], requires_grad=True):
        origin = [[i["a"], i["alpha"], i["d"], i["delta"]] for i in l]
        return tf.tensor(np.array(origin, dtype=np.float), requires_grad=requires_grad)

    def __len__(self):
        return len(self.joints)

    def __getitem__(self, key):
        return self.joints[key]

    def dump(self):
        return [j.dump() for j in self.joints]

    def load(self, obj):
        for i, j in enumerate(self.joints):
            j.load(obj[i])

    def __sub__(self, other: "KinematicChain"):
        assert len(self.joints) == len(other.joints)
        return KinematicChain([self.joints[i] - other.joints[i] for i in range(len(self.joints))])

    def __repr__(self) -> str:
        return "KinematicChain([\n"+",\n".join(map(lambda x: "\t"+repr(x), self.joints))+"\n])"

    def transform(self, thetas: List[float]):
        assert len(thetas) >= len(self.joints)
        T = tf.as_tensor(np.eye(4), dtype=tf.float64)
        for j, theta in zip(self.joints, thetas):
            T = T@j.transform(theta)
        return T

    def display(self, ax, thetas):
        assert len(thetas) >= len(self.joints)
        T = tf.as_tensor(np.eye(4), dtype=tf.float64)
        p = np.array([0, 0, 0, 1])
        lines = []

        def plot(a, b):
            return ax.plot(
                [a[0]/a[3], b[0]/b[3]],
                [a[1]/a[3], b[1]/b[3]],
                [a[2]/a[3], b[2]/b[3]]
            )
        for j in range(len(self.joints)):
            T = T@self.joints[j].transform(thetas[j])
            q = T.detach().numpy()@np.array([0, 0, 0, 1])
            x = T.detach().numpy()@np.array([10, 0, 0, 1])
            y = T.detach().numpy()@np.array([0, 10, 0, 1])
            z = T.detach().numpy()@np.array([0, 0, 10, 1])
            lines.extend(plot(p, q))
            lines.extend(plot(x, q))
            lines.extend(plot(y, q))
            lines.extend(plot(z, q))
            p = q
        return lines

if __name__ == "__main__":
    from .robots import kuka_lbr_iiwa_7
    param = KinematicChain.obj_to_tensor(kuka_lbr_iiwa_7(), requires_grad=True)
    chain = KinematicChain.from_tensor(param)
    print(chain)
