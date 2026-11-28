import os
import torch

ENABLE_CUDA_GRAPH = os.environ.get("ENABLE_CUDA_GRAPH", "True").lower() == "true"
# 采用"单一 master buffer + 多 bucket graph"的模式时，必须预先分配最大 batch，
# 否则在 capture 之后扩容/重新分配会导致 graph 记录的地址失效。
CUDA_GRAPH_MAX_BS = int(os.environ.get("CUDA_GRAPH_MAX_BS", "128"))
_SHAPE_GUARD_FASTPATH = (
    os.environ.get("CUDAGRAPH_DISABLE_SHAPE_GUARD_FASTPATH", "0") != "1"
)


class CUDAGraph_Wrapper:
    """
    支持"分桶（bucket）"的 CUDA Graph Wrapper：
    - batch size 变化时，按 bucket_size 选择/复用不同的 CUDAGraph
    - 对于 batch_size < bucket_size：把输入 copy 到 static buffer 的 [:B] 区间，replay 后返回输出的 [:B]
    """

    def __init__(
        self,
        model,
        warm_up_times: int = 3,
        enable: bool | None = None,
        batch_size_key: str | None = None,
    ):
        self.model = model
        self.warm_up_times = warm_up_times
        self._enable = ENABLE_CUDA_GRAPH if enable is None else bool(enable)
        self._batch_size_key = batch_size_key or "suffix_inputs_embeds"

        # graph cache: bucket_bs -> CUDAGraph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # 对每个 bucket，保存对 master buffer 的 view（graph 捕获依赖这些 view 的地址/shape/stride）
        self.static_inputs_map: dict[int, dict[str, torch.Tensor]] = {}
        self.static_output_tensor: dict[int, torch.Tensor] = {}
        self.graph_pool = None

        # master buffers（所有 bucket 共用同一块）
        self.graph_vars: dict[str, torch.Tensor] = {}
        self._max_bs: int | None = None

        # 记录"非 batch 维度形状签名"（全局一致），避免静默错误
        self._shape_signature: dict[str, tuple] = {}
        # 记录每个输入的 batch 维度位置（默认 0；例如 suffix_position_ids 的 batch 在 dim=1）
        self._batch_dim: dict[str, int] = {}
        self._shape_guard_verified: dict[int, bool] = {}

    def _get_batch_size(self, **kwargs) -> int:
        key = self._batch_size_key
        if key in kwargs:
            v = kwargs[key]
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                return int(v.shape[0])
        raise ValueError(f"CUDAGraph_Wrapper.forward 需要 {key} 参数")

    def _select_bucket_bs(self, bs: int) -> int:
        """
        分桶策略：
        - 1,2,4,8 走 pow2
        - >8 时按 16 对齐（16,32,48,...）
        """
        if bs <= 1:
            return 1
        if bs <= 2:
            return 2
        if bs <= 4:
            return 4
        if bs <= 8:
            return 8
        return int(((bs + 15) // 16) * 16)

    def _batch_dim_for_key(self, key: str, tensor: torch.Tensor) -> int:
        # 目前仅对已知的特殊输入做最小特判
        # suffix_position_ids: shape [3, B, T]，batch 在 dim=1
        if key == "suffix_position_ids" and tensor.dim() >= 2:
            return 1
        return 0

    def _view_for_bucket(self, tensor: torch.Tensor, bucket_bs: int, batch_dim: int):
        if tensor.dim() == 0:
            return tensor
        if batch_dim == 0:
            return tensor[:bucket_bs]
        if batch_dim == 1:
            return tensor[:, :bucket_bs]
        raise ValueError(f"Unsupported batch_dim={batch_dim} for cudagraph wrapper")

    def _copy_and_pad(
        self,
        master: torch.Tensor,
        tensor: torch.Tensor,
        bs: int,
        bucket_bs: int,
        batch_dim: int,
    ):
        if tensor.dim() == 0:
            master.copy_(tensor)
            return
        if batch_dim == 0:
            master[:bs].copy_(tensor)
            if bs < bucket_bs:
                master[bs:bucket_bs].zero_()
            return
        if batch_dim == 1:
            master[:, :bs].copy_(tensor)
            if bs < bucket_bs:
                master[:, bs:bucket_bs].zero_()
            return
        raise ValueError(f"Unsupported batch_dim={batch_dim} for cudagraph wrapper")

    def _allocate_master_tensor(
        self, key: str, tensor: torch.Tensor, max_bs: int, bs: int
    ) -> torch.Tensor:
        """
        为某个输入分配 master buffer（单一大 buffer）：
        - 只扩展 batch 维度到 max_bs，其余维度保持不变
        - 先拷贝当前 bs 的有效区间，其余位置初始化为 0（避免 replay 时读到旧值）
        """
        if tensor.dim() == 0:
            return tensor.clone()
        batch_dim = self._batch_dim_for_key(key, tensor)
        shape = list(tensor.shape)
        if batch_dim >= len(shape):
            raise ValueError(
                f"Invalid batch_dim={batch_dim} for key={key}, tensor.shape={tuple(tensor.shape)}"
            )
        shape[batch_dim] = max_bs
        buf = torch.zeros(tuple(shape), device=tensor.device, dtype=tensor.dtype)
        # 初始 copy：pad 到当前 bucket（至少覆盖 bs 的有效区间）
        self._copy_and_pad(buf, tensor, bs=bs, bucket_bs=bs, batch_dim=batch_dim)
        return buf

    def _ensure_master_buffers(self, bs: int, **kwargs):
        """
        确保 master buffers 已分配（只分配一次）。
        - max_bs 来自环境变量 CUDA_GRAPH_MAX_BS；如果未设置，则使用第一次看到的 bucket_bs 作为 max_bs，
          并禁止后续超过该值（否则 graph 会因扩容而失效）。
        """
        if self._max_bs is not None:
            return

        bucket_bs = self._select_bucket_bs(bs)
        max_bs = CUDA_GRAPH_MAX_BS if CUDA_GRAPH_MAX_BS > 0 else bucket_bs
        self._max_bs = int(max_bs)

        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(
                    f"CUDAGraph_Wrapper 仅支持 Tensor kwargs，当前 {k}={type(v)}"
                )
            batch_dim = self._batch_dim_for_key(k, v) if v.dim() > 0 else 0
            self._batch_dim[k] = batch_dim
            # 记录非 batch 维度（全局一致）
            if v.dim() == 0:
                self._shape_signature[k] = tuple()
            else:
                sig = list(v.shape)
                sig.pop(batch_dim)
                self._shape_signature[k] = tuple(sig)
            self.graph_vars[k] = self._allocate_master_tensor(
                k, v, max_bs=self._max_bs, bs=bs
            )

    def _initialize_bucket(self, bucket_bs: int, bs: int, **kwargs):
        assert self._max_bs is not None
        self._shape_guard_verified[bucket_bs] = False
        if bucket_bs > self._max_bs:
            raise ValueError(
                f"CUDAGraph bucket_bs({bucket_bs}) 超过 master max_bs({self._max_bs}). "
                f"请设置环境变量 CUDA_GRAPH_MAX_BS >= {bucket_bs}。"
            )

        print(f"[CUDAGraph] Initializing bucket_bs={bucket_bs} (current bs={bs}) ...")

        # 生成该 bucket 的 view map（固定地址/shape/stride）
        static_map: dict[str, torch.Tensor] = {}
        for k, v in kwargs.items():
            # 形状 guard（非 batch 维度必须一致）
            expected = self._shape_signature.get(k)
            if v.dim() == 0:
                got = tuple()
            else:
                batch_dim = self._batch_dim.get(k, self._batch_dim_for_key(k, v))
                got_list = list(v.shape)
                got_list.pop(batch_dim)
                got = tuple(got_list)
            if expected is not None and got != expected:
                raise ValueError(
                    f"CUDAGraph 初始化 bucket 时遇到非 batch 维度变化: key={k}, expected={expected}, got={got}"
                )
            master = self.graph_vars[k]
            batch_dim = self._batch_dim.get(k, 0)
            static_map[k] = self._view_for_bucket(
                master, bucket_bs=bucket_bs, batch_dim=batch_dim
            )

        # warmup（让内核/缓存稳定）
        out = None
        for _ in range(self.warm_up_times):
            out = self.forward_naive(**static_map)
        assert isinstance(out, torch.Tensor)

        # output master buffer（只分配一次）
        if "_outputs" not in self.graph_vars:
            out_shape = (self._max_bs,) + tuple(out.shape[1:])
            self.graph_vars["_outputs"] = torch.empty(
                out_shape, device=out.device, dtype=out.dtype
            )

        self.static_inputs_map[bucket_bs] = static_map
        self.static_output_tensor[bucket_bs] = self.graph_vars["_outputs"][:bucket_bs]

        # capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            self.static_output_tensor[bucket_bs].copy_(self.forward_naive(**static_map))
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bucket_bs] = graph
        torch.cuda.synchronize()
        print(f"[CUDAGraph] Bucket {bucket_bs} captured.")

    def forward_naive(self, **kwargs):
        return self.model(**kwargs)

    def forward(self, **kwargs):
        if not self._enable:
            return self.forward_naive(**kwargs)
        bs = self._get_batch_size(**kwargs)
        bucket_bs = self._select_bucket_bs(bs)

        # 初始化 master buffers（第一次调用）
        self._ensure_master_buffers(bs=bs, **kwargs)
        assert self._max_bs is not None
        if bucket_bs > self._max_bs:
            # 不允许运行时超过 master max_bs（会导致需要扩容，从而使已捕获 graph 失效）
            return self.forward_naive(**kwargs)

        # lazy capture for this bucket
        if bucket_bs not in self.graphs:
            self._initialize_bucket(bucket_bs=bucket_bs, bs=bs, **kwargs)

        # shape guard：非 batch 维度变化时，cudagraph 不安全，直接 fallback
        # fast-path: skip guard if already verified for this bucket (stable-state optimization)
        if not (
            _SHAPE_GUARD_FASTPATH and self._shape_guard_verified.get(bucket_bs, False)
        ):
            for k, v in kwargs.items():
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"CUDAGraph_Wrapper 仅支持 Tensor kwargs，当前 {k}={type(v)}"
                    )
                expected = self._shape_signature.get(k)
                if v.dim() == 0:
                    got = tuple()
                else:
                    batch_dim = self._batch_dim.get(k, self._batch_dim_for_key(k, v))
                    got_list = list(v.shape)
                    got_list.pop(batch_dim)
                    got = tuple(got_list)
                if expected is not None and got != expected:
                    return self.forward_naive(**kwargs)
            # Mark verified for fast-path on subsequent calls
            self._shape_guard_verified[bucket_bs] = True

        # copy inputs into master buffer（按各自 batch_dim 写入，并清 0 padding）
        for k, tensor in kwargs.items():
            master = self.graph_vars[k]
            batch_dim = self._batch_dim.get(k, 0)
            self._copy_and_pad(
                master, tensor, bs=bs, bucket_bs=bucket_bs, batch_dim=batch_dim
            )

        graph = self.graphs[bucket_bs]
        graph.replay()
        return self.graph_vars["_outputs"][:bs]
