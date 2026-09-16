#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-06-27 18:32:23
Author: jly
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr, 
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,

):
    # from 0 to the left of diag
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # diag block where the causal mask is not obvious ?
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    # non causal
    else:
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # bc we deal with k transpose so seq len is axis1
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV) # triton optimization, see later

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block) # k already transposed bc loaded with stride inverted

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :]) # offset matrix of size (block_size_q, block_size_kv) with broadcast when comparing
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) # axis 1 bc one max per row within a block
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) * softmax_scale
            QK_block = QK_block * softmax_scale - m_ij[:, None]


        P_block = tl.math.exp(QK_block)

        l_ij = tl.sum(P_block, 1) # normalisaiton factor of the current block

        alpha = tl.math.exp(m_i - m_ij) # correction factor

        # correct the normalisation factgor and add the current one
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None] # correct the output
        O_block = tl.dot(P_block, V_block, O_block) # third paramter is accumulator O_block += P_block @ V_block
        # its an optimiztion bt dot product is a sum so add it dirrectly into the accumulator

        m_i = m_ij

        # advance to next block of k and v
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i
            
@triton.jit  
def _attn_fwd(
    Q, #B, num_heads, seq_len, head_dim
    K,
    V, 
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq, 
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
): 
    
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM) 

    block_index_q = tl.program_id(0)

    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch 
        + index_head.to(tl.int64) * stride_Q_head
    )

    # the idea is to parallelize over batch and head but also over seq len
    # thats why we split up seq len in seq_len / block_size 

    """
    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, # points exactaly to the beginning of the target batch, head
        shape=(SEQ_LEN, HEAD_DIM), # at this location, we treat an array of this shape (not really it will be the block shape)
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # queries to skip (thoses that the process dont use)
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # the shape that the process tackle
        order=(1, 0), # idk : .T ?
    )

    # for v we only parallise it over batch and head, not on seq len
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_head, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # we want k transpose for dot product
    K_block_ptr = tl.make_block_ptr(
        base= K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_head), #to transpsoe swqp the stride 
        offsets=(0,0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0,1),
    )


    O_block_ptr = tl.make_block_ptr(
        base= O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_head, stride_O_dim), #to transpsoe swqp the stride 
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1,0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    offs_kv = tl.arange(0,  BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the block of Q to SRAM
    Q_block = tl.load(Q_block_ptr)


    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )


    if STAGE == 3:
        O_block, l_i, m_i = -_attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )
    

    m_i += tl.math.log(
        l_i
    ) # needed to compute the logsumexp for the backward pass
    # when compute exp xi - mi iy will be exp xi -mi / li , but why needded ?

    
    O_block = O_block / l_i[:, None]

    
    
    return 

class TritonAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape


        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q) # K and V are not necessarily the same shape as Q, eg. CrossAttention
        stage = 3 if causal else 1



        # ceil(seq_len/blocksize_q)
        # we can parallize over block Qi bc in O, each row are for diffrent block of queries
        # grid of instances
        # in triton, us the user, dont specify the mapping thread by thread but by group of thread
        # triton itself does the mapping thread by thread (in cuda we have to do it ourselves)
        # x , y, z
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # whivh group of queries are we going to work with?
            BATCH_SIZE * NUM_HEADS, # we paralize at batch level but also at multi head level and then by blocks of query
            1, # Z in the CUDA launch grid
        )

        # nb of kernels = batch_size * num_heads * num_blocks_q

        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        
        
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3), 
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

def test_op(
        BATCH_SIZE: int,
        NUM_HEADS: int,
        SEQ_LEN: int, 
        HEAD_DIM: int,
        causal: bool,
        dtype: float = torch.float16,
):
    
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=.0, std=.5)
        .requires_grad_()
    )

    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=.0, std=.5)
        .requires_grad_()
    )

    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=.0, std=.5)
        .requires_grad_()
    )


    softmax_scale = 1 / (HEAD_DIM**0.5)

    d0 = torch.randn_like(Q) # for backward pass?

    # to not see the future
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))

    # K shape: (B, num_heads, seq_len, head_dim )
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    # -> seq_len, seq_len
    if causal:
        P[:, :, MASK==0] = float("-inf")

    P = torch.softmax(P.float(), dim=-1).half() # to fp16

    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0)

    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    tri_out = TrittonAttention.apply(Q, K, V, causal, softmax_scale).half()

    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None


    rtol = 0.0 
    atol = 1e-2

    assert torch.allclose(ref_0, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    

   