package ggml

import "core:c"

// Compile with:
// mkdir build
// cd build
// cmake .. -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
// cmake --build . --config Release

MAX_DIMS :: 4
MAX_PARAMS :: 2048
MAX_CONTEXTS :: 64
MAX_SRC :: 10
MAX_NAME :: 64
MAX_OP_PARAMS :: 64
DEFAULT_N_THREADS :: 4
DEFAULT_GRAPH_SIZE :: 2048
FILE_MAGIC :: 0x67676d6c

type :: enum c.int {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    COUNT,
}

op :: enum c.int {
    NONE = 0,
    DUP,
    ADD,
    ADD1,
    ACC,
    SUB,
    MUL,
    DIV,
    SQR,
    SQRT,
    LOG,
    SUM,
    SUM_ROWS,
    MEAN,
    ARGMAX,
    REPEAT,
    REPEAT_BACK,
    CONCAT,
    SILU_BACK,
    NORM,
    RMS_NORM,
    RMS_NORM_BACK,
    GROUP_NORM,
    MUL_MAT,
    MUL_MAT_ID,
    OUT_PROD,
    SCALE,
    SET,
    CPY,
    CONT,
    RESHAPE,
    VIEW,
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    GET_ROWS_BACK,
    DIAG,
    DIAG_MASK_INF,
    DIAG_MASK_ZERO,
    SOFT_MAX,
    SOFT_MAX_BACK,
    ROPE,
    ROPE_BACK,
    CLAMP,
    CONV_TRANSPOSE_1D,
    IM2COL,
    CONV_TRANSPOSE_2D,
    POOL_1D,
    POOL_2D,
    UPSCALE,
    PAD,
    ARANGE,
    TIMESTEP_EMBEDDING,
    ARGSORT,
    LEAKY_RELU,
    FLASH_ATTN_EXT,
    FLASH_ATTN_BACK,
    SSM_CONV,
    SSM_SCAN,
    WIN_PART,
    WIN_UNPART,
    GET_REL_POS,
    ADD_REL_POS,
    UNARY,
    MAP_UNARY,
    MAP_BINARY,
    MAP_CUSTOM1_F32,
    MAP_CUSTOM2_F32,
    MAP_CUSTOM3_F32,
    MAP_CUSTOM1,
    MAP_CUSTOM2,
    MAP_CUSTOM3,
    CROSS_ENTROPY_LOSS,
    CROSS_ENTROPY_LOSS_BACK,
    COUNT,
}

unary_op :: enum c.int {
    ABS,
    SGN,
    NEG,
    STEP,
    TANH,
    ELU,
    RELU,
    SIGMOID,
    GELU,
    GELU_QUICK,
    SILU,
    HARDSWISH,
    HARDSIGMOID,
    COUNT,
}

log_level :: enum c.int {
    ERROR = 2,
    WARN  = 3,
    INFO  = 4,
    DEBUG = 5,
}

backend_type :: enum c.int {
    CPU = 0,
    GPU = 10,
    GPU_SPLIT = 20,
}

ggml_context :: struct{}

backend_t :: rawptr
backend_buffer :: struct{}
backend_buffer_t :: ^backend_buffer

init_params :: struct {
    mem_size: c.size_t,
    mem_buffer: rawptr,
    no_alloc: bool,
}

tensor :: struct {
    type: type,
    backend: backend_type, // Deprecated
    buffer: ^backend_buffer,
    ne: [MAX_DIMS]c.int64_t,
    nb: [MAX_DIMS]c.size_t,
    op: op,
    op_params: [MAX_OP_PARAMS / size_of(c.int32_t)]c.int32_t,
    flags: c.int32_t,
    grad: ^tensor,
    src: [MAX_SRC]^tensor,
    view_src: ^tensor,
    view_offs: c.size_t,
    data: rawptr,
    name: [MAX_NAME]c.char,
    extra: rawptr,
    padding: [4]c.char,
}

object_type :: enum c.int {
    TENSOR,
    GRAPH,
    WORK_BUFFER,
}

object :: struct {
    offs: c.size_t,
    size: c.size_t,
    next: ^object,
    type: object_type,
    padding: [4]c.char,
}

OBJECT_SIZE :: size_of(object)

cgraph_eval_order :: enum c.int {
    LEFT_TO_RIGHT = 0,
    RIGHT_TO_LEFT,
    COUNT,
}

hash_set :: struct {
    size: c.size_t,
    keys: [^]^tensor,
}

cgraph :: struct {
    size: c.int,
    n_nodes: c.int,
    n_leafs: c.int,
    nodes: [^]^tensor,
    grads: [^]^tensor,
    leafs: [^]^tensor,
    visited_hash_table: hash_set,
    order: cgraph_eval_order,
}

abort_callback :: #type proc "c" (data: rawptr) -> bool

cplan :: struct {
    work_size: c.size_t,
    work_data: [^]c.uint8_t,
    n_threads: c.int,
    abort_callback: abort_callback,
    abort_callback_data: rawptr,
}

status :: enum c.int {
    ALLOC_FAILED = -2,
    FAILED = -1,
    SUCCESS = 0,
    ABORTED = 1,
}

opt_type :: enum c.int {
    ADAM,
    LBFGS,
}

linesearch :: enum c.int {
    DEFAULT = 1,
    BACKTRACKING_ARMIJO = 0,
    BACKTRACKING_WOLFE = 1,
    BACKTRACKING_STRONG_WOLFE = 2,
}

opt_result :: enum c.int {
    OK = 0,
    DID_NOT_CONVERGE,
    NO_CONTEXT,
    INVALID_WOLFE,
    FAIL,
    CANCEL,
    LINESEARCH_FAIL = -128,
    LINESEARCH_MINIMUM_STEP,
    LINESEARCH_MAXIMUM_STEP,
    LINESEARCH_MAXIMUM_ITERATIONS,
    LINESEARCH_INVALID_PARAMETERS,
}

opt_callback :: #type proc "c" (data: rawptr, accum_step: c.int, sched: ^f32, cancel: ^bool)
log_callback :: #type proc "c" (level: log_level, text: cstring, user_data: rawptr)

opt_params :: struct {
    type: opt_type,
    graph_size: c.size_t,
    n_threads: c.int,
    past: c.int,
    delta: f32,
    max_no_improvement: c.int,
    print_forward_graph: c.bool,
    print_backward_graph: c.bool,
    n_gradient_accumulation: c.int,
    adam: struct {
        n_iter: c.int,
        sched: f32,
        decay: f32,
        decay_min_ndim: c.int,
        alpha: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        eps_f: f32,
        eps_g: f32,
        gclip: f32,
    },
    lbfgs: struct {
        m: c.int,
        n_iter: c.int,
        max_linesearch: c.int,
        eps: f32,
        ftol: f32,
        wolfe: f32,
        min_step: f32,
        max_step: f32,
        linesearch: linesearch,
    }
}

opt_context :: struct {
    ctx: ^ggml_context,
    params: opt_params,
    iter: c.int,
    nx: c.int64_t,
    just_initialized: bool,
    loss_before: f32,
    loss_after: f32,
    adam: struct {
        g: ^tensor,
        m: ^tensor,
        v: ^tensor,
        pf: ^tensor,
        fx_best: f32,
        fx_prev: f32,
        n_no_improvement: c.int,
    },
    lbfgs: struct {
        x: ^tensor,
        xp: ^tensor,
        g: ^tensor,
        gp: ^tensor,
        d: ^tensor,
        pf: ^tensor,
        lmal: ^tensor,
        lmys: ^tensor,
        lms: ^tensor,
        lmy: ^tensor,
        fx_best: f32,
        step: f32,
        j: c.int,
        k: c.int,
        end: c.int,
        n_no_improvement: c.int,
    },
}

tallocr :: struct {
    buffer: backend_buffer_t,
    base: rawptr,
    alignment: c.size_t,
    offset: c.size_t,
}

gallocr_t :: rawptr
backend_buffer_type_t :: rawptr

@(extra_linker_flags="/NODEFAULTLIB:libcmt")
foreign import ggml {"ggml.lib"}
// foreign import ggml {"ggml.lib", "cuda.lib", "cudart.lib", "cublas.lib"}

@(default_calling_convention="c", link_prefix="ggml_")
foreign ggml {
    init :: proc(params: init_params) -> ^ggml_context ---
    free :: proc(ctx: ^ggml_context) ---

    build_forward_expand :: proc(cgraph: ^cgraph, t: ^tensor) ---
    build_backward_expand :: proc(ctx: ^ggml_context, gf, gb: ^cgraph, keep: bool) ---

    graph_plan :: proc(cgraph: ^cgraph, n_threads: c.int) -> cplan ---
    graph_compute :: proc(cgraph: ^cgraph, cplan: ^cplan) -> status ---
    graph_compute_with_ctx :: proc(ctx: ^ggml_context, cgraph: ^cgraph, n_threads: c.int) -> status ---

    new_graph :: proc(ctx: ^ggml_context) -> ^cgraph ---
    new_graph_custom :: proc(ctx: ^ggml_context, size: c.size_t, grads: bool) -> ^cgraph ---
    graph_dup :: proc(ctx: ^ggml_context, g: ^cgraph) -> ^cgraph ---
    graph_view :: proc(g: ^cgraph, i0, i1: c.int) -> cgraph ---
    graph_cpy :: proc(src, dst: ^cgraph) ---
    graph_reset :: proc(g: ^cgraph) ---
    graph_clear :: proc(g: ^cgraph) ---

    graph_dump_dot :: proc(gb, gf: ^cgraph, filename: cstring) ---

    new_tensor :: proc(ctx: ^ggml_context, type: type, n_dims: c.int, ne: ^c.int64_t) -> ^tensor ---
    new_tensor_1d :: proc(ctx: ^ggml_context, type: type, ne0: c.int64_t) -> ^tensor ---
    new_tensor_2d :: proc(ctx: ^ggml_context, type: type, ne0, ne1: c.int64_t) -> ^tensor ---
    new_tensor_3d :: proc(ctx: ^ggml_context, type: type, ne0, ne1, ne2: c.int64_t) -> ^tensor ---
    new_tensor_4d :: proc(ctx: ^ggml_context, type: type, ne0, ne1, ne2, ne3: c.int64_t) -> ^tensor ---

    tensor_overhead :: proc() -> c.size_t ---
    graph_overhead :: proc() -> c.size_t ---
    graph_overhead_custom :: proc(size: c.size_t, grads: bool) -> c.size_t ---
    used_mem :: proc(ctx: ^ggml_context) -> c.size_t ---

    type_size :: proc(type: type) -> c.size_t ---

    n_dims :: proc(t: ^tensor) -> c.int ---
    nelements :: proc(t: ^tensor) -> c.int64_t ---

    set_param :: proc(ctx: ^ggml_context, t: ^tensor) ---

    set_f32 :: proc(t: ^tensor, value: f32) -> ^tensor ---
    set_f32_1d :: proc(t: ^tensor, i: c.int, value: f32) ---
    set_f32_nd :: proc(t: ^tensor, i0, i1, i2, i3: c.int, value: f32) ---

    get_f32_1d :: proc(t: ^tensor, i: c.int) -> f32 ---
    get_f32_nd :: proc(t: ^tensor, i0, i1, i2, i3: c.int) -> f32 ---

    get_data_f32 :: proc(t: ^tensor) -> [^]f32 ---

    get_name :: proc(t: ^tensor) -> cstring ---
    set_name :: proc(t: ^tensor, name: cstring) -> ^tensor ---

    mul :: proc(ctx: ^ggml_context, a, b: ^tensor) -> ^tensor ---
    add :: proc(ctx: ^ggml_context, a, b: ^tensor) -> ^tensor ---

    mul_mat :: proc(ctx: ^ggml_context, a, b: ^tensor) -> ^tensor ---

    relu :: proc(ctx: ^ggml_context, a: ^tensor) -> ^tensor ---
    soft_max :: proc(ctx: ^ggml_context, a: ^tensor) -> ^tensor ---

    repeat :: proc(ctx: ^ggml_context, a, b: ^tensor) -> ^tensor ---

    cross_entropy_loss :: proc(ctx: ^ggml_context, a, b: ^tensor) -> ^tensor ---

    nbytes :: proc(t: ^tensor) -> c.size_t ---

    opt_default_params :: proc(type: opt_type) -> opt_params ---
    opt :: proc(ctx: ^ggml_context, params: opt_params, f: ^tensor) -> opt_result ---

    opt_init :: proc(ctx: ^ggml_context, opt: ^opt_context, params: opt_params, nx: c.int64_t) ---
    opt_resume :: proc(ctx: ^ggml_context, opt: ^opt_context, f: ^tensor) -> opt_result ---
    opt_resume_g :: proc(ctx: ^ggml_context, opt: ^opt_context, f: ^tensor, gf, gb: ^cgraph, callback: opt_callback, callback_data: rawptr) -> opt_result ---

    set_input :: proc(t: ^tensor) ---
    set_output :: proc(t: ^tensor) ---

    backend_cpu_init :: proc() -> backend_t ---
    backend_alloc_buffer :: proc(backend: backend_t, size: c.size_t) -> backend_buffer_t ---
    backend_buffer_get_type :: proc(buffer: backend_buffer_t) -> backend_buffer_type_t ---
    backend_graph_compute :: proc(backend: backend_t, graph: ^cgraph) -> status ---

    tallocr_new :: proc(buffer: backend_buffer_t) -> tallocr ---
    tallocr_alloc :: proc(talloc: ^tallocr, t: ^tensor) ---

    gallocr_new :: proc(buf: backend_buffer_type_t) -> gallocr_t ---
    gallocr_new_n :: proc(bufts: [^]backend_buffer_type_t, n_bufs: c.int) -> gallocr_t ---
    gallocr_free :: proc(galloc: gallocr_t) ---
    gallocr_reserve :: proc(galloc: gallocr_t, graph: ^cgraph) -> bool ---
    gallocr_alloc_graph :: proc(galloc: gallocr_t, graph: ^cgraph) -> bool ---

    backend_free :: proc(backend: backend_t) ---
    backend_buffer_free :: proc(buffer: backend_buffer_t) ---

    backend_alloc_ctx_tensors_from_buft :: proc(ctx: ^ggml_context, buft: backend_buffer_type_t) -> ^backend_buffer ---
    backend_alloc_ctx_tensors :: proc(ctx: ^ggml_context, backend: backend_t) -> ^backend_buffer ---

    backend_get_default_buffer_type :: proc(backend: backend_t) -> backend_buffer_type_t ---
    backend_cpu_buffer_type :: proc() -> backend_buffer_type_t ---
    backend_cuda_init :: proc(device: c.int) -> backend_t ---

    backend_is_cpu :: proc(backend: backend_t) -> bool ---
    backend_cpu_set_n_threads :: proc(backend_cpu: backend_t, n_threads: c.int) ---

    backend_tensor_set :: proc(t: ^tensor, data: rawptr, offset, size: c.size_t) ---
    backend_tensor_get :: proc(t: ^tensor, data: rawptr, offset, size: c.size_t) ---
}