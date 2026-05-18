from __future__ import annotations

import torch
from typing import Tuple


def _ensure_cpu(t: torch.Tensor) -> Tuple[torch.Tensor, torch.device]:
    dev = t.device
    if dev.type == "cpu":
        return t, dev
    return t.cpu(), dev


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # q: (...,4) in (w,x,y,z)
    q_cpu, dev = _ensure_cpu(q)
    conj = q_cpu.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj.to(dev)


def quat_inv(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # safe inverse: conjugate / max(norm^2, eps)
    q_cpu, dev = _ensure_cpu(q)
    norm2 = (q_cpu * q_cpu).sum(dim=-1, keepdim=True)
    denom = norm2.clamp(min=eps)
    inv = quat_conjugate(q_cpu) / denom
    return inv.to(dev)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Multiply quaternions a * b (w,x,y,z)
    a_cpu, dev = _ensure_cpu(a)
    b_cpu, _ = _ensure_cpu(b)
    w1, x1, y1, z1 = a_cpu.unbind(dim=-1)
    w2, x2, y2, z2 = b_cpu.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = torch.stack([w, x, y, z], dim=-1)
    return out.to(dev)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Rotate vector v by quaternion q. q (...,4), v (...,3)
    q_cpu, dev = _ensure_cpu(q)
    v_cpu, _ = _ensure_cpu(v)
    # q * [0, v] * q^{-1}
    q_inv = quat_inv(q_cpu)
    v_as_quat = torch.cat([torch.zeros_like(v_cpu[..., :1]), v_cpu], dim=-1)
    res = quat_mul(quat_mul(q_cpu, v_as_quat), q_inv)
    return res[..., 1:].to(dev)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    # Return rotation matrix (...,3,3) from quaternion (w,x,y,z)
    q_cpu, dev = _ensure_cpu(q)
    # normalize
    norm = q_cpu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    qn = q_cpu / norm
    w, x, y, z = qn.unbind(dim=-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz
    mat = torch.stack([torch.stack([m00, m01, m02], dim=-1), torch.stack([m10, m11, m12], dim=-1), torch.stack([m20, m21, m22], dim=-1)], dim=-2)
    return mat.to(dev)


def subtract_frame_transforms(
    origin_pos: torch.Tensor, origin_quat: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute target expressed in origin frame: returns (pos, quat)
    # Move to CPU for safe numeric ops
    op_cpu, dev = _ensure_cpu(origin_pos)
    oq_cpu, _ = _ensure_cpu(origin_quat)
    tp_cpu, _ = _ensure_cpu(target_pos)
    tq_cpu, _ = _ensure_cpu(target_quat)

    # delta pos
    delta = tp_cpu - op_cpu
    pos_b = quat_apply(oq_cpu, delta)
    # orientation in origin frame
    ori_b = quat_mul(quat_inv(oq_cpu), tq_cpu)
    return pos_b.to(dev), ori_b.to(dev)
