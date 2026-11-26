import torch
import torch.distributed as dist

from falcon.distributed import get_cp_group

dist_all_gather_func = dist.all_gather_into_tensor
dist_reduce_scatter_func = dist.reduce_scatter_tensor


class _GatherFromContextParallelRegion(torch.autograd.Function):
    """Gather the input from context parallel region and concatinate."""

    @staticmethod
    def symbolic(
        graph,
        input_,
        tensor_parallel_output_grad: bool = True,
        group=None,
        output_split_sizes=None,
    ):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(input_, group, output_split_sizes)

    @staticmethod
    def forward(
        ctx,
        input_,
        tensor_parallel_output_grad: bool = True,
        group=None,
        output_split_sizes=None,
    ):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        return _gather_along_first_dim(input_, group, output_split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(grad_output, ctx.group, ctx.output_split_sizes),
                None,
                None,
                None,
                None,
            )
        else:
            assert ctx.output_split_sizes is None
            return _split_along_first_dim(grad_output, ctx.group), None, None, None, None


def _gather_along_first_dim(input_, group=None, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (list[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """

    if group is None:
        group = get_cp_group()
    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        dist.all_gather(output_tensor_list, input_, group=group)

    return output


def _reduce_scatter_along_first_dim(input_, group=None, input_split_sizes=None):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    if group is None:
        group = get_cp_group()

    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        dist_reduce_scatter_func(output, input_.contiguous(), group=group)
    else:
        rank = dist.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        output = torch.empty_like(input_tensor_list[rank])
        dist.reduce_scatter(output, input_tensor_list, group=group)
    return output


def _split_along_first_dim(input_, group=None):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    if group is None:
        group = get_cp_group()
    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = dist.get_rank(group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def gather_from_context_parallel_region(
    input_,
    tensor_parallel_output_grad=False,
    group=None,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _GatherFromContextParallelRegion.apply(input_, tensor_parallel_output_grad, group, output_split_sizes)
