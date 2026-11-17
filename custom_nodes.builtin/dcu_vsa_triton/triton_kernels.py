from __future__ import annotations

import contextlib
import math

import torch
import triton
import triton.language as tl
from comfy_kitchen._rope_utils import check_rope_inplace, detect_rms_rope_bnhd

BLOCK_SIZE = 64
HEAD_DIM = 128
_LOG2E = 1.4426950408889634
_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _partial_rms_rope_qk_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_scale_ptr,
    k_scale_ptr,
    q_out_ptr,
    k_out_ptr,
    q_heads,
    head_dim,
    rot_dim,
    freqs_batch,
    freqs_seq,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_qod,
    stride_kob,
    stride_koh,
    stride_kos,
    stride_kod,
    stride_fb,
    stride_fs,
    stride_fd,
    stride_fr,
    stride_fp,
    epsilon,
    compute_dtype: tl.constexpr,
    norm_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROT_HALF: tl.constexpr,
    HAS_PARTIAL_TAIL: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_slot = tl.program_id(1)
    seq_idx = tl.program_id(2)

    is_k = head_slot >= q_heads
    head_idx = tl.where(is_k, head_slot - q_heads, head_slot)

    if is_k:
        x_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh + seq_idx * stride_ks
        out_base = k_out_ptr + batch_idx * stride_kob + head_idx * stride_koh + seq_idx * stride_kos
        s_ptr = k_scale_ptr
        stride_xd = stride_kd
        stride_od = stride_kod
    else:
        x_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + seq_idx * stride_qs
        out_base = q_out_ptr + batch_idx * stride_qob + head_idx * stride_qoh + seq_idx * stride_qos
        s_ptr = q_scale_ptr
        stride_xd = stride_qd
        stride_od = stride_qod

    full_offs = tl.arange(0, BLOCK_D)
    full_mask = full_offs < head_dim
    x_full = tl.load(x_base + full_offs * stride_xd, mask=full_mask, other=0.0).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)

    if HAS_PARTIAL_TAIL:
        scale_full = tl.load(s_ptr + full_offs, mask=full_mask, other=0.0).to(tl.float32)
        normed_full = (x_full * inv_rms * scale_full).to(norm_dtype).to(out_dtype)
        tail_mask = full_mask & (full_offs >= rot_dim)
        tl.store(out_base + full_offs * stride_od, normed_full, mask=tail_mask)

    half_rot = rot_dim // 2
    rot_offs = tl.arange(0, BLOCK_ROT_HALF)
    rot_mask = rot_offs < half_rot

    x_0 = tl.load(x_base + rot_offs * stride_xd, mask=rot_mask, other=0.0).to(tl.float32)
    x_1 = tl.load(x_base + (rot_offs + half_rot) * stride_xd, mask=rot_mask, other=0.0).to(tl.float32)
    s_0 = tl.load(s_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    s_1 = tl.load(s_ptr + rot_offs + half_rot, mask=rot_mask, other=0.0).to(tl.float32)

    x_0 = (x_0 * inv_rms * s_0).to(norm_dtype).to(compute_dtype)
    x_1 = (x_1 * inv_rms * s_1).to(norm_dtype).to(compute_dtype)

    fb_idx = tl.where(freqs_batch == 1, 0, batch_idx)
    fs_idx = tl.where(freqs_seq == 1, 0, seq_idx)
    f_base = freqs_ptr + fb_idx * stride_fb + fs_idx * stride_fs + rot_offs * stride_fd

    f00 = tl.load(f_base, mask=rot_mask, other=0.0)
    f01 = tl.load(f_base + stride_fp, mask=rot_mask, other=0.0)
    f10 = tl.load(f_base + stride_fr, mask=rot_mask, other=0.0)
    f11 = tl.load(f_base + stride_fr + stride_fp, mask=rot_mask, other=0.0)

    out_0 = (f00 * x_0 + f01 * x_1).to(out_dtype)
    out_1 = (f10 * x_0 + f11 * x_1).to(out_dtype)

    tl.store(out_base + rot_offs * stride_od, out_0, mask=rot_mask)
    tl.store(out_base + (rot_offs + half_rot) * stride_od, out_1, mask=rot_mask)


def triton_rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused RMSNorm + split-half RoPE for both full and partial rot_dim."""
    if k_scale is None:
        k_scale = q_scale
    if inplace:
        check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    if not q_scale.is_contiguous():
        q_scale = q_scale.contiguous()
    if not k_scale.is_contiguous():
        k_scale = k_scale.contiguous()

    head_dim = q.shape[3]
    eff_rot = rot_dim if rot_dim else head_dim
    bnhd = detect_rms_rope_bnhd(q, freqs_cis, rot_dim=eff_rot)
    if bnhd is None:
        raise ValueError("RMS-RoPE frequencies are not broadcastable to input")

    batch = q.shape[0]
    if bnhd:
        seq_len, q_heads = q.shape[1], q.shape[2]
        k_heads = k.shape[2]
        s_qh, s_qs = q.stride(2), q.stride(1)
        s_kh, s_ks = k.stride(2), k.stride(1)
        stride_fs = freqs_cis.stride(1)
        freqs_seq = freqs_cis.shape[1]
    else:
        q_heads, seq_len = q.shape[1], q.shape[2]
        k_heads = k.shape[1]
        s_qh, s_qs = q.stride(1), q.stride(2)
        s_kh, s_ks = k.stride(1), k.stride(2)
        stride_fs = freqs_cis.stride(2)
        freqs_seq = freqs_cis.shape[2]

    if inplace:
        q_out, k_out = q, k
    else:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

    s_qoh, s_qos = (q_out.stride(2), q_out.stride(1)) if bnhd else (q_out.stride(1), q_out.stride(2))
    s_koh, s_kos = (k_out.stride(2), k_out.stride(1)) if bnhd else (k_out.stride(1), k_out.stride(2))

    block_d = triton.next_power_of_2(head_dim)
    block_rot_half = max(16, triton.next_power_of_2(eff_rot // 2))
    grid = (batch, q_heads + k_heads, seq_len)

    _partial_rms_rope_qk_kernel[grid](
        q,
        k,
        freqs_cis,
        q_scale,
        k_scale,
        q_out,
        k_out,
        q_heads,
        head_dim,
        eff_rot,
        freqs_cis.shape[0],
        freqs_seq,
        q.stride(0), s_qh, s_qs, q.stride(3),
        k.stride(0), s_kh, s_ks, k.stride(3),
        q_out.stride(0), s_qoh, s_qos, q_out.stride(3),
        k_out.stride(0), s_koh, s_kos, k_out.stride(3),
        freqs_cis.stride(0), stride_fs, freqs_cis.stride(3), freqs_cis.stride(4), freqs_cis.stride(5),
        epsilon,
        compute_dtype=_DTYPE_MAP.get(freqs_cis.dtype, tl.float32),
        norm_dtype=_DTYPE_MAP.get(q.dtype, tl.bfloat16),
        out_dtype=_DTYPE_MAP.get(q.dtype, tl.bfloat16),
        BLOCK_D=block_d,
        BLOCK_ROT_HALF=block_rot_half,
        HAS_PARTIAL_TAIL=(eff_rot < head_dim),
        num_warps=4,
    )
    return q_out, k_out


@triton.jit
def _chunk_qkv_rope_bhnd_kernel(
    chunk_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    freqs_ptr,
    qw_ptr,
    kw_ptr,
    block_len_ptr,
    t0,
    heads,
    rot_dim,
    freqs_seq,
    stride_cm,
    stride_qh,
    stride_qt,
    stride_fs,
    stride_fd,
    stride_fr,
    stride_fp,
    epsilon,
    compute_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    HAS_BLOCK_LEN: tl.constexpr,
    HAS_PARTIAL_TAIL: tl.constexpr,
    BLOCK_ROT_HALF: tl.constexpr,
):
    head_idx = tl.program_id(0)
    local_t = tl.program_id(1)
    global_t = t0 + local_t

    d_offs = tl.arange(0, 128)
    inner = heads * 128

    q_dst = q_out_ptr + head_idx * stride_qh + global_t * stride_qt + d_offs
    k_dst = k_out_ptr + head_idx * stride_qh + global_t * stride_qt + d_offs
    v_dst = v_out_ptr + head_idx * stride_qh + global_t * stride_qt + d_offs

    if HAS_BLOCK_LEN:
        blk_idx = global_t // 64
        valid_len = tl.load(block_len_ptr + blk_idx)
        if (global_t % 64) >= valid_len:
            zeros = tl.zeros([128], dtype=out_dtype)
            tl.store(q_dst, zeros)
            tl.store(k_dst, zeros)
            tl.store(v_dst, zeros)
            return

    row_base = chunk_ptr + local_t * stride_cm
    v_src = tl.load(row_base + 2 * inner + head_idx * 128 + d_offs)
    tl.store(v_dst, v_src.to(out_dtype))

    q_base = row_base + head_idx * 128
    k_base = row_base + inner + head_idx * 128

    q_full = tl.load(q_base + d_offs).to(tl.float32)
    k_full = tl.load(k_base + d_offs).to(tl.float32)

    q_inv = tl.math.rsqrt(tl.sum(q_full * q_full, axis=0) / 128.0 + epsilon)
    k_inv = tl.math.rsqrt(tl.sum(k_full * k_full, axis=0) / 128.0 + epsilon)

    if HAS_PARTIAL_TAIL:
        qw_full = tl.load(qw_ptr + d_offs).to(tl.float32)
        kw_full = tl.load(kw_ptr + d_offs).to(tl.float32)
        tail_mask = d_offs >= rot_dim
        tl.store(q_dst, (q_full * q_inv * qw_full).to(out_dtype), mask=tail_mask)
        tl.store(k_dst, (k_full * k_inv * kw_full).to(out_dtype), mask=tail_mask)

    half_rot = rot_dim // 2
    rot_offs = tl.arange(0, BLOCK_ROT_HALF)
    rot_mask = rot_offs < half_rot

    q0 = tl.load(q_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    q1 = tl.load(q_base + rot_offs + half_rot, mask=rot_mask, other=0.0).to(tl.float32)
    k0 = tl.load(k_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    k1 = tl.load(k_base + rot_offs + half_rot, mask=rot_mask, other=0.0).to(tl.float32)

    qw0 = tl.load(qw_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    qw1 = tl.load(qw_ptr + rot_offs + half_rot, mask=rot_mask, other=0.0).to(tl.float32)
    kw0 = tl.load(kw_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    kw1 = tl.load(kw_ptr + rot_offs + half_rot, mask=rot_mask, other=0.0).to(tl.float32)

    q0 = (q0 * q_inv * qw0).to(out_dtype).to(compute_dtype)
    q1 = (q1 * q_inv * qw1).to(out_dtype).to(compute_dtype)
    k0 = (k0 * k_inv * kw0).to(out_dtype).to(compute_dtype)
    k1 = (k1 * k_inv * kw1).to(out_dtype).to(compute_dtype)

    fs_idx = tl.where(freqs_seq == 1, 0, global_t)
    f_base = freqs_ptr + fs_idx * stride_fs + rot_offs * stride_fd
    f00 = tl.load(f_base, mask=rot_mask, other=0.0)
    f01 = tl.load(f_base + stride_fp, mask=rot_mask, other=0.0)
    f10 = tl.load(f_base + stride_fr, mask=rot_mask, other=0.0)
    f11 = tl.load(f_base + stride_fr + stride_fp, mask=rot_mask, other=0.0)

    q_out_base = q_out_ptr + head_idx * stride_qh + global_t * stride_qt
    k_out_base = k_out_ptr + head_idx * stride_qh + global_t * stride_qt

    tl.store(q_out_base + rot_offs, (f00 * q0 + f01 * q1).to(out_dtype), mask=rot_mask)
    tl.store(q_out_base + rot_offs + half_rot, (f10 * q0 + f11 * q1).to(out_dtype), mask=rot_mask)
    tl.store(k_out_base + rot_offs, (f00 * k0 + f01 * k1).to(out_dtype), mask=rot_mask)
    tl.store(k_out_base + rot_offs + half_rot, (f10 * k0 + f11 * k1).to(out_dtype), mask=rot_mask)


@triton.jit
def _compact_exact_mask_kernel(
    mask_ptr,
    out_idx_ptr,
    n_blocks,
    max_routed,
    BLOCK_C: tl.constexpr,
):
    row_id = tl.program_id(0)
    mask_row = mask_ptr + row_id * n_blocks
    out_row = out_idx_ptr + row_id * max_routed
    written = 0
    offs_base = tl.arange(0, BLOCK_C)
    for start in range(0, n_blocks, BLOCK_C):
        cols = start + offs_base
        in_bounds = cols < n_blocks
        flag = tl.load(mask_row + cols, mask=in_bounds, other=0).to(tl.int32)
        prefix = tl.cumsum(flag, axis=0)
        dst = written + prefix - 1
        tl.store(out_row + dst, cols, mask=in_bounds & (flag != 0))
        written += tl.sum(flag, axis=0)


@triton.jit
def _sol_attn_triton_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lengths_ptr,
    routed_idx_ptr,
    routed_cnt_ptr,
    exact_mask_ptr,
    km_ptr,
    vm_ptr,
    log2_len_ptr,
    kb_ptr,
    t_len,
    n_blocks,
    heads,
    sink_kv0,
    sink_kv1,
    sink_q0,
    sink_q1,
    log2s,
    max_routed,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_kbb,
    USE_TAIL: tl.constexpr,
    HAS_KEY_BIAS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    qi = tl.program_id(1)
    b_idx = bh_idx // heads
    h_idx = bh_idx % heads

    offs_m = qi * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_D)
    q_mask = offs_m < t_len

    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh

    q_tile = tl.load(
        q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_D], dtype=tl.float32)
    local_n = tl.arange(0, BLOCK_N)

    is_sink_q = (qi >= sink_q0) & (qi < sink_q1)
    n_sink_kv = sink_kv1 - sink_kv0
    row_slot = bh_idx * n_blocks + qi

    if is_sink_q:
        total_exact = n_blocks
    else:
        total_exact = n_sink_kv + tl.load(routed_cnt_ptr + row_slot)

    routed_row_ptr = routed_idx_ptr + row_slot * max_routed

    # Single unified exact loop (eliminates 3x duplicate tl.dot live ranges & VGPR spills)
    for idx in range(0, total_exact):
        need_routed = (~is_sink_q) & (idx >= n_sink_kv)
        routed_pos = tl.where(need_routed, idx - n_sink_kv, 0)
        routed_kj = tl.load(routed_row_ptr + routed_pos, mask=need_routed, other=0)
        kj = tl.where(is_sink_q, idx, tl.where(idx < n_sink_kv, sink_kv0 + idx, routed_kj))

        k_len = tl.load(lengths_ptr + kj)
        k_valid = local_n < k_len
        offs_n = kj * BLOCK_N + local_n

        k_tile_t = tl.load(
            k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kt,
            mask=k_valid[None, :],
            other=0.0,
        )
        s_tok = tl.dot(q_tile, k_tile_t).to(tl.float32) * log2s
        if HAS_KEY_BIAS:
            kb_vals = tl.load(kb_ptr + b_idx * stride_kbb + offs_n, mask=k_valid, other=0.0)
            s_tok = s_tok + kb_vals[None, :]
        s_tok = tl.where(k_valid[None, :], s_tok, -1.0e30)

        m_blk = tl.max(s_tok, axis=1)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s_tok - m_new[:, None])
        p = tl.where(k_valid[None, :], p, 0.0)

        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=k_valid[:, None],
            other=0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(COMPUTE_DTYPE), v_tile, acc * alpha[:, None])
        m_i = m_new

    # Pooled tail compensation (Sol-Attn) for non-sink query rows
    if USE_TAIL:
        if not is_sink_q:
            km_base = km_ptr + bh_idx * n_blocks * HEAD_D
            vm_base = vm_ptr + bh_idx * n_blocks * HEAD_D
            em_row_ptr = exact_mask_ptr + row_slot * n_blocks

            for p_start in range(0, n_blocks, BLOCK_N):
                p_offs = p_start + local_n
                p_in_bounds = p_offs < n_blocks
                is_exact = tl.load(em_row_ptr + p_offs, mask=p_in_bounds, other=1)
                tail_valid = p_in_bounds & (is_exact == 0)

                km_tile_t = tl.load(
                    km_base + offs_d[:, None] * 1 + p_offs[None, :] * HEAD_D,
                    mask=p_in_bounds[None, :],
                    other=0.0,
                )
                log2_len_tile = tl.load(log2_len_ptr + p_offs, mask=p_in_bounds, other=0.0)

                s_blk = tl.dot(q_tile, km_tile_t).to(tl.float32) * log2s + log2_len_tile[None, :]
                s_blk = tl.where(tail_valid[None, :], s_blk, -1.0e30)

                m_blk = tl.max(s_blk, axis=1)
                m_new = tl.maximum(m_i, m_blk)
                alpha = tl.math.exp2(m_i - m_new)
                p_tail = tl.math.exp2(s_blk - m_new[:, None])
                p_tail = tl.where(tail_valid[None, :], p_tail, 0.0)

                vm_tile = tl.load(
                    vm_base + p_offs[:, None] * HEAD_D + offs_d[None, :],
                    mask=p_in_bounds[:, None],
                    other=0.0,
                )
                l_i = l_i * alpha + tl.sum(p_tail, axis=1)
                acc = tl.dot(p_tail.to(COMPUTE_DTYPE), vm_tile, acc * alpha[:, None])
                m_i = m_new

    out_tile = acc / tl.maximum(l_i, 1.0e-30)[:, None]
    tl.store(
        o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        out_tile.to(COMPUTE_DTYPE),
        mask=q_mask[:, None],
    )


def _block_lengths(t: int, n: int, device: torch.device, block_len: torch.Tensor | None) -> torch.Tensor:
    tail_len = t - (n - 1) * BLOCK_SIZE
    if block_len is not None:
        lengths = block_len.to(device=device, dtype=torch.int32).clamp(1, BLOCK_SIZE)
        if tail_len < BLOCK_SIZE:
            lengths = lengths.clone()
            lengths[-1] = lengths[-1].clamp(max=tail_len)
        return lengths
    lengths = torch.full((n,), BLOCK_SIZE, dtype=torch.int32, device=device)
    if tail_len < BLOCK_SIZE:
        lengths[-1] = tail_len
    return lengths


def _normalize_key_bias(key_bias: torch.Tensor, batch: int, t: int, device: torch.device) -> torch.Tensor:
    if key_bias.device != device:
        raise ValueError(f"sol_attn: key_bias must be on {device}, got {key_bias.device}")
    if key_bias.dim() == 4:
        if key_bias.shape[1] != 1 or key_bias.shape[2] != 1:
            raise ValueError(f"sol_attn: key_bias must be key-only, got {tuple(key_bias.shape)}")
        key_bias = key_bias[:, 0, 0, :]
    if key_bias.dim() == 1:
        key_bias = key_bias.unsqueeze(0)
    if key_bias.dim() != 2 or key_bias.shape[-1] != t or key_bias.shape[0] not in (1, batch):
        raise ValueError(f"sol_attn: unexpected key_bias shape {tuple(key_bias.shape)} for B={batch}, T={t}")
    if key_bias.dtype == torch.bool:
        key_bias = torch.where(key_bias, 0.0, float("-inf"))
    return (key_bias.float() * _LOG2E).expand(batch, t).contiguous()


def _pool_block_means(
    x: torch.Tensor,
    n_blocks: int,
    lengths: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute [B*H, N, D] float32 means over live rows from [B, T, H, D] or strided view."""
    b, t, h, d = x.shape
    pad = n_blocks * BLOCK_SIZE - t
    # Permute to [B, H, T, D] so block reduction along T is contiguous
    xb = x.permute(0, 2, 1, 3)
    if valid_mask is not None:
        xb = xb * valid_mask.view(1, 1, t, 1)
    if pad:
        xb = torch.cat([xb, xb.new_zeros(b, h, pad, d)], dim=2)
    sums = xb.reshape(b * h, n_blocks, BLOCK_SIZE, d).sum(dim=2, dtype=torch.float32)
    return sums / lengths.to(torch.float32).view(1, n_blocks, 1)


def _compute_routing_and_coarse(
    qm: torch.Tensor,
    km: torch.Tensor,
    vm: torch.Tensor,
    n_blocks: int,
    tau: float,
    scale: float,
    topk_ratio: float,
    sink_kv0: int,
    sink_kv1: int,
    sink_q0: int,
    sink_q1: int,
    need_coarse: bool,
):
    """Compute block routing table and optional VSA coarse output from [BH, N, D] fp32 means."""
    log2s = scale * _LOG2E
    scores_nat = torch.bmm(qm, km.transpose(1, 2)) * scale
    oc = torch.bmm(torch.softmax(scores_nat, dim=-1), vm) if need_coarse else None

    k_mean = km.mean(dim=1, keepdim=True)
    kcc = km - k_mean
    # colmean[qi, kj] == qm[qi] @ kcc[kj]^T * log2s == (scores_nat[qi, kj] - (qm[qi] @ k_mean^T) * scale) * _LOG2E
    shift = torch.bmm(qm, k_mean.transpose(1, 2)) * scale
    colmean = (scores_nat - shift) * _LOG2E

    if topk_ratio > 0.0:
        ranked = colmean.half()
        if sink_kv1 > sink_kv0:
            ranked[:, :, sink_kv0:sink_kv1] = float("-inf")
        sink_cnt = max(0, min(sink_kv1, n_blocks) - min(sink_kv0, n_blocks))
        rem = n_blocks - sink_cnt
        kk = max(0, min(rem - 1, max(1, round(topk_ratio * rem)))) if rem > 1 else 0
        if kk > 0:
            row_thr = ranked.topk(kk, dim=-1).values[..., -1:]
            exact = ranked >= row_thr
        else:
            exact = torch.zeros_like(ranked, dtype=torch.bool)
    else:
        kc_var = kcc.pow(2).mean(dim=1, keepdim=True)
        var = (qm.pow(2) * kc_var).sum(dim=-1, keepdim=True)
        thr = tau * torch.sqrt(var * (log2s * log2s) + 1e-6)
        exact = colmean > thr

    idx = torch.arange(n_blocks, device=qm.device)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).unsqueeze(0)
    if sink_kv1 > sink_kv0:
        exact[:, :, sink_kv0:sink_kv1] = True
    if sink_q1 > sink_q0:
        exact[:, sink_q0:sink_q1, :] = True

    # Build compact per-row block list excluding sink_kv (handled inline) and sink_q (dense inline)
    routed_exact = exact.clone()
    if sink_kv1 > sink_kv0:
        routed_exact[:, :, sink_kv0:sink_kv1] = False
    if sink_q1 > sink_q0:
        routed_exact[:, sink_q0:sink_q1, :] = False

    routed_cnt = routed_exact.sum(dim=-1, dtype=torch.int32).contiguous()
    max_routed = int(routed_cnt.max().item()) if routed_cnt.numel() > 0 else 0
    if max_routed > 0:
        routed_idx = torch.empty((qm.shape[0], n_blocks, max_routed), dtype=torch.int32, device=qm.device)
        routed_u8 = routed_exact.to(torch.uint8).contiguous()
        _compact_exact_mask_kernel[(qm.shape[0] * n_blocks,)](
            routed_u8,
            routed_idx,
            n_blocks,
            max_routed,
            BLOCK_C=1024,
            num_warps=4,
        )
    else:
        max_routed = 1
        routed_idx = torch.empty((qm.shape[0], n_blocks, 1), dtype=torch.int32, device=qm.device)

    return exact, routed_idx, routed_cnt, max_routed, oc


def _add_coarse_(out: torch.Tensor, oc: torch.Tensor, gate: torch.Tensor) -> None:
    """In-place out += gate * oc[block(t)], matching comfy_kitchen eager/sol_attn.py."""
    b, t, h, d = out.shape
    oc = oc.view(b, h, -1, d).permute(0, 2, 1, 3).contiguous()
    gate = gate.contiguous()
    nfull = t // BLOCK_SIZE
    if nfull:
        out[:, :nfull * BLOCK_SIZE].view(b, nfull, BLOCK_SIZE, h, d).addcmul_(
            gate[:, :nfull * BLOCK_SIZE].view(b, nfull, BLOCK_SIZE, h, d),
            oc[:, :nfull, None],
        )
    if t % BLOCK_SIZE:
        out[:, nfull * BLOCK_SIZE:].addcmul_(gate[:, nfull * BLOCK_SIZE:], oc[:, nfull:nfull + 1])


def _run_sol_attn_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    qm: torch.Tensor,
    km: torch.Tensor,
    vm: torch.Tensor,
    lengths: torch.Tensor,
    tau: float,
    scale: float,
    sink_blocks: list[int] | None,
    sink_q: list[int] | None,
    key_bias: torch.Tensor | None,
    topk_ratio: float,
    tail: bool,
    coarse_gate: torch.Tensor | None,
) -> torch.Tensor:
    b, t, h, d = q.shape
    n_blocks = lengths.numel()
    bh = b * h
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]
    sink_kv0 = max(0, min(sink_kv0, n_blocks))
    sink_kv1 = max(sink_kv0, min(sink_kv1, n_blocks))
    sink_q0 = max(0, min(sink_q0, n_blocks))
    sink_q1 = max(sink_q0, min(sink_q1, n_blocks))

    exact, routed_idx, routed_cnt, max_routed, oc = _compute_routing_and_coarse(
        qm, km, vm, n_blocks, tau, scale, topk_ratio,
        sink_kv0, sink_kv1, sink_q0, sink_q1,
        need_coarse=(coarse_gate is not None),
    )

    if tail:
        exact_mask = exact.to(torch.uint8).contiguous()
        km_cast = km.to(q.dtype).contiguous()
        vm_cast = vm.to(q.dtype).contiguous()
        log2_len = torch.log2(lengths.to(torch.float32)).contiguous()
    else:
        exact_mask = torch.empty((1,), dtype=torch.uint8, device=q.device)
        km_cast = torch.empty((1,), dtype=q.dtype, device=q.device)
        vm_cast = torch.empty((1,), dtype=q.dtype, device=q.device)
        log2_len = torch.empty((1,), dtype=torch.float32, device=q.device)

    kb = _normalize_key_bias(key_bias, b, t, q.device) if key_bias is not None else torch.empty((1,), dtype=torch.float32, device=q.device)
    stride_kbb = kb.stride(0) if key_bias is not None else 0

    grid = (bh, n_blocks)
    amd_knobs = getattr(getattr(triton, "knobs", None), "amd", None)
    if amd_knobs is not None:
        ctx = amd_knobs.scope()
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if amd_knobs is not None:
            amd_knobs.use_in_thread_transpose = True
        _sol_attn_triton_kernel[grid](
            q, k, v, out,
            lengths, routed_idx, routed_cnt,
            exact_mask, km_cast, vm_cast, log2_len, kb,
            t, n_blocks, h,
            sink_kv0, sink_kv1, sink_q0, sink_q1,
            scale * _LOG2E, max_routed,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_kbb,
            USE_TAIL=bool(tail),
            HAS_KEY_BIAS=(key_bias is not None),
            COMPUTE_DTYPE=_DTYPE_MAP[q.dtype],
            BLOCK_M=BLOCK_SIZE,
            BLOCK_N=BLOCK_SIZE,
            HEAD_D=HEAD_DIM,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=16,
            kpack=2,
        )

    if coarse_gate is not None and oc is not None:
        _add_coarse_(out, oc, coarse_gate)
    return out


def triton_sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
) -> torch.Tensor:
    """Triton Sol-Attn / VSA implementation matching comfy_kitchen contract on gfx936."""
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"triton_sol_attn requires bfloat16 or float16, got {q.dtype}")
    b, t, h, d = q.shape
    if d != HEAD_DIM:
        raise ValueError(f"triton_sol_attn requires head_dim={HEAD_DIM}, got {d}")
    if scale is None:
        scale = d ** -0.5

    n_blocks = (t + BLOCK_SIZE - 1) // BLOCK_SIZE
    lengths = _block_lengths(t, n_blocks, q.device, block_len)
    if block_len is not None:
        pos = torch.arange(t, device=q.device)
        valid_mask = ((pos % BLOCK_SIZE) < lengths[pos // BLOCK_SIZE]).to(q.dtype)
    else:
        valid_mask = None

    qm = _pool_block_means(q, n_blocks, lengths, valid_mask)
    km = _pool_block_means(k, n_blocks, lengths, valid_mask)
    vm = _pool_block_means(v, n_blocks, lengths, valid_mask)

    out = torch.empty((b, t, h, d), dtype=q.dtype, device=q.device)
    return _run_sol_attn_core(
        q, k, v, out, qm, km, vm, lengths,
        tau=float(tau), scale=float(scale),
        sink_blocks=sink_blocks, sink_q=sink_q,
        key_bias=key_bias, topk_ratio=float(topk_ratio),
        tail=bool(tail), coarse_gate=coarse_gate,
    )


def triton_sol_attn_chunked(
    qkv_chunks,
    t: int,
    h: int,
    rope_freqs: torch.Tensor,
    qk_norm_weights: tuple[torch.Tensor, torch.Tensor],
    kmean: torch.Tensor | None = None,
    vscale: torch.Tensor | None = None,
    tau: float = 1.0,
    topk_ratio: float = 0.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    rope_eps: float = 1e-6,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
):
    """Chunked-producer Triton Sol-Attn / VSA for MiniMax-H3 on gfx936."""
    rot = rope_freqs.shape[-3] * 2
    if rot % 2 or not 0 < rot <= HEAD_DIM:
        raise ValueError(f"sol_attn_chunked: invalid rot_dim {rot}")
    dev = rope_freqs.device
    if scale is None:
        scale = HEAD_DIM ** -0.5

    n_blocks = (t + BLOCK_SIZE - 1) // BLOCK_SIZE
    lengths = _block_lengths(t, n_blocks, dev, block_len)
    qw = qk_norm_weights[0].to(device=dev, dtype=torch.bfloat16).contiguous()
    kw = qk_norm_weights[1].to(device=dev, dtype=torch.bfloat16).contiguous()

    # Allocate BHND contiguous tensors so both chunk write and attention load are coalesced
    q_bhnd = torch.empty((1, h, t, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    k_bhnd = torch.empty((1, h, t, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    v_bhnd = torch.empty((1, h, t, HEAD_DIM), dtype=torch.bfloat16, device=dev)

    has_block_len = block_len is not None
    len_arg = lengths if has_block_len else torch.empty((1,), dtype=torch.int32, device=dev)
    block_rot_half = max(16, triton.next_power_of_2(rot // 2))
    width = 3 * h * HEAD_DIM

    t0 = 0
    iterable = qkv_chunks() if callable(qkv_chunks) else qkv_chunks
    for chunk in iterable:
        m = chunk.shape[-2]
        if m == 0:
            continue
        if t0 % BLOCK_SIZE:
            raise ValueError("sol_attn_chunked: chunk starts must be 64-aligned")
        if chunk.shape[-1] != width or chunk.dtype != torch.bfloat16 or chunk.device != dev:
            raise ValueError(
                f"sol_attn_chunked: expected [M, {width}] bfloat16 on {dev}, got {tuple(chunk.shape)} {chunk.dtype}"
            )
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()
        _chunk_qkv_rope_bhnd_kernel[(h, m)](
            chunk,
            q_bhnd, k_bhnd, v_bhnd,
            rope_freqs, qw, kw, len_arg,
            t0, h, rot, rope_freqs.shape[1],
            chunk.stride(0),
            q_bhnd.stride(1), q_bhnd.stride(2),
            rope_freqs.stride(1), rope_freqs.stride(3), rope_freqs.stride(4), rope_freqs.stride(5),
            float(rope_eps),
            compute_dtype=_DTYPE_MAP[rope_freqs.dtype],
            out_dtype=tl.bfloat16,
            HAS_BLOCK_LEN=has_block_len,
            HAS_PARTIAL_TAIL=(rot < HEAD_DIM),
            BLOCK_ROT_HALF=block_rot_half,
            num_warps=1,
        )
        t0 += m
    if t0 != t:
        raise ValueError(f"sol_attn_chunked: chunks cover {t0} tokens, expected {t}")

    # Dead rows were already zeroed during chunk scatter, so pooling reduces directly
    pad = n_blocks * BLOCK_SIZE - t
    def _pool_bhnd(xb: torch.Tensor) -> torch.Tensor:
        if pad:
            xb = torch.cat([xb, xb.new_zeros(1, h, pad, HEAD_DIM)], dim=2)
        return xb.reshape(h, n_blocks, BLOCK_SIZE, HEAD_DIM).sum(dim=2, dtype=torch.float32) / lengths.to(torch.float32).view(1, n_blocks, 1)

    qm = _pool_bhnd(q_bhnd)
    km = _pool_bhnd(k_bhnd)
    vm = _pool_bhnd(v_bhnd)

    # View as [1, T, H, 128] strided without copying; kernel reads arbitrary strides directly
    q_view = q_bhnd.transpose(1, 2)
    k_view = k_bhnd.transpose(1, 2)
    v_view = v_bhnd.transpose(1, 2)
    out = torch.empty((1, t, h, HEAD_DIM), dtype=torch.bfloat16, device=dev)

    _run_sol_attn_core(
        q_view, k_view, v_view, out,
        qm, km, vm, lengths,
        tau=float(tau), scale=float(scale),
        sink_blocks=sink_blocks, sink_q=sink_q,
        key_bias=None, topk_ratio=float(topk_ratio),
        tail=bool(tail), coarse_gate=coarse_gate,
    )

    dummy_stat = torch.zeros((h, HEAD_DIM), dtype=torch.float32, device=dev)
    return out, dummy_stat, dummy_stat
