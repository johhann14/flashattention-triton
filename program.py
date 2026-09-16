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

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
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
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # we want k transpose for dot product
    K_block_ptr = tl.make_block_ptr(
        base= K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq), #to transpsoe swqp the stride 
        offsets=(0,0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0,1),
    )


    O_block_ptr = tl.make_block_ptr(
        base= O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim), #to transpsoe swqp the stride 
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


    m_ptrs= M + index_batch_head * SEQ_LEN + offs_q 
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
    
    return 

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D, # [BS, NUM_HEADS, SEQ_LEN]
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
): 
    block_index_q = tl.program_id(0) # axis 0 of the grid used to launch the kernel
    # here is the index of the block 

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of O

    # start from O ptr, move from index batch had seq 
    # should be [BLOCK_SIZE_Q, HEAD_DIM]c
    # O shape is BS, num_heads, seq_len, head_dim 
    # REMINDER: we parallize over batch, head, and q blocks
    # # with hardcoded offset like this, it supposes that the array is contiguous  
    O_block = tl.load(
        O 
        + index_batch_head * HEAD_DIM * SEQ_LEN 
        + offs_q[:, None] * HEAD_DIM # broadcast bc of this + below -> [BLOCK_SIZE_Q, HEAD_DIM]
        + offs_dim[None, :]
    )
    # so O_block is a 2D array of value that comes from a 2D array pointer
    # that represents the adress of each element this program/kernel handle 

    dO_block = tl.load(
        dO 
        + index_batch_head * HEAD_DIM * SEQ_LEN 
        + offs_q[:, None] * HEAD_DIM # broadcast bc of this + below -> [BLOCK_SIZE_Q, HEAD_DIM]
        + offs_dim[None, :]
    ).to(tl.float32)

    D_block = tl.sum(dO_block * O_block, axis=1) # shape (Block_size_q) : see paper for the formula

    # need to store

    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q

    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M, # log sum exp so we dont need to recompute the max and normalization factor
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):

    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # i think same thing as using the offset but here take account if the object is transpose or no, so the stride will be transpose if needed and so its more robuts?
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)


    # offsezt_batch_head_seq = (index_batch * stride_batch) + (index_head * stride_head) + (offs_q *stride_seq)
    # stride_batch = num_head * seq_len
    # stride_head = seq_len
    # stride_seq = 1
    # offset_batch_head_seq = (index_batch * num_head * seq_len) + (index_head * seq_len) + off_q
    # = (index_batch * num_head + index_head) * seq_len + off_q
    # = index_batch_head * seq_len + off_q
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # move to pointer to the right batch head
    
    Q+= offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M+= offset_batch_head_seq
    D+= offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    #block_kv is th esizer of the block of lkv which is a block of token 
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0,BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # 2D array of size [block_kv, head_dim  ]
    # k is laready moves to the correct batch and head, and from there we load the 2D array of seq_len head dim 
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim  
    )
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim  
    )

    offs_q = tl.arange(0, BLOCK_Q)

    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    # it will load a [HEAD_dim, SEQ_len] array which is the Q.T
    # it works bc we have a 2D array of adresses and each adresses at ij loads qji 
    # offs q broadcatst along the row
    # offs dum along the column 
    # so the adress at ij correspondds to the adresse of qji

    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q

    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs) 
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)



        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m [None, :]) # to reeally compute the softmax of ST, to have PT

        if STAGE == 3:
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            ) # true = not masked: shape [BLOCK_KV, BLOCK_Q]

            P_T_block = tl.where(mask_block, P_T_block, 0.0) # we can mask after softmax bc normalisation has already be computed by taking the mask 

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block).to(tl.float32)

        Di= tl.load(D+offs_q)

        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q* stride_seq


    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M, # log sum exp so we dont need to recompute the max and normalization factor
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # i think same thing as using the offset but here take account if the object is transpose or no, so the stride will be transpose if needed and so its more robuts?
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)


    # offsezt_batch_head_seq = (index_batch * stride_batch) + (index_head * stride_head) + (offs_q *stride_seq)
    # stride_batch = num_head * seq_len
    # stride_head = seq_len
    # stride_seq = 1
    # offset_batch_head_seq = (index_batch * num_head * seq_len) + (index_head * seq_len) + off_q
    # = (index_batch * num_head + index_head) * seq_len + off_q
    # = index_batch_head * seq_len + off_q
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # move to pointer to the right batch head
    
    Q+= offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M+= offset_batch_head_seq
    D+= offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_q = tl.program_id(0)

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    M_block = tl.load(M + offs_q) 
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV

    for blk_index in range(num_steps):

        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # auto refressive masking

            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    tl.store(dQ_block_ptrs, dQ_block)

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

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)


        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)

        D = torch.empty_like(M) # [BS, NUM_HEDS, SEQ_lEN]


        # compute Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M, # log sum exp so we dont need to recompute the max and normalization factor
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )         

        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M, # log sum exp so we dont need to recompute the max and normalization factor
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,      
        )

        return dQ, dK, dV, None, None

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

    dO = torch.randn_like(Q) # for backward pass?

    # to not see the future
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))

    # K shape: (B, num_heads, seq_len, head_dim )
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    # -> seq_len, seq_len
    if causal:
        P[:, :, MASK==0] = float("-inf")

    P = torch.softmax(P.float(), dim=-1).half() # to fp16

    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)

    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None


    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None


    rtol = 0.0 
    atol = 1e-2

    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    
    print(f"All assert passed!")
    

if __name__ == "__main__":
    test_op(BATCH_SIZE=2, NUM_HEADS=8, SEQ_LEN=1024, HEAD_DIM=32, causal=True)
    test_op(BATCH_SIZE=2, NUM_HEADS=8, SEQ_LEN=1024, HEAD_DIM=32, causal=False)    
   