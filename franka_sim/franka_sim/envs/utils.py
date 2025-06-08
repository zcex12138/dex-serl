import numpy as np
from scipy.spatial.transform import Rotation
# import jax.numpy as jnp


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(abs(x))

def symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def quat_conjugate(q):
    """JAX实现四元数共轭 (batch支持)"""
    # q shape: (...,4)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)

def quat_mul(q1, q2):
    """JAX实现批量四元数乘法 (Hamilton积)"""
    # q1, q2 shape: (...,4)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.concatenate([w, x, y, z], axis=-1)

def rotation_distance(object_rot, target_rot):
    """Compute the rotation distance between two quaternions."""
    # Compute the conjugate of the target rotation
    target_rot_conjugate = quat_conjugate(target_rot)
    # Compute the relative rotation
    relative_rotation = quat_mul(object_rot, target_rot_conjugate)
    # Compute the angle of the relative rotation
    angle = 2.0 * np.arccos(np.clip(relative_rotation[0], -1.0, 1.0))
    return angle

def random_quat():
    """生成一个随机四元数"""
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    # 生成一个随机角度
    angle = np.random.uniform(0, 2 * np.pi)
    # 计算四元数
    q = np.array([np.cos(angle / 2), *(vec * np.sin(angle / 2))])
    return q

def mtx2quat(R):
    r = Rotation.from_matrix(R)
    q = r.as_quat() # 返回顺序为(x, y, z, w)
    # 转换为(w, x, y, z)格式
    return np.array([q[3], q[0], q[1], q[2]])

def rpy2mtx(r,p,y):
    # 内旋XYZ顺序
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
    # 组合旋转矩阵
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def rpy2quat(r, p, y):
    R = rpy2mtx(r, p, y)
    q = mtx2quat(R)
    return q