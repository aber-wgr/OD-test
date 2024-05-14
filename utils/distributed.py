import torch
import torch.distributed as dist
import os

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    #os.environ["NCCL_DEBUG"] = "INFO"

    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    print("GPUs per Node:" + str(ngpus_per_node))

    # GPU count is a bit complicated here. We have ngpus_per_node GPUs per node, and a number of local tasks that need to share them.
    # We need to figure out what our local rank is, and from there determine which GPUs we can use.

    if args.distributed:
        if 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
            print("running LOCAL_RANK mode")
            args.rank = int(os.environ["RANK"])
            args.gpu = int(os.environ['LOCAL_RANK'])
            # calculate the gpulist for this process (for example, if we have 4 GPUs and 2 processes, the first process will get GPUs 0 and 1, the second 2 and 3)
            args.gpulist = list("cuda:" + str(p) for p in range(args.gpu * ngpus_per_node, (args.gpu + 1) * ngpus_per_node))

            print("Rank " + str(args.rank) + " GPUs:" + str(args.gpulist))
        elif 'SLURM_PROCID' in os.environ:
            print("running SLURM_PROCID mode")
            args.rank = int(os.environ['SLURM_PROCID'])
            local_processes = int(os.environ['SLURM_NTASKS_PER_NODE'])
            args.gpu = args.rank % local_processes

            gpus_per_process = ngpus_per_node // local_processes
            args.gpulist = list("cuda:" + str(p) for p in range(args.gpu * gpus_per_process, (args.gpu + 1) * gpus_per_process))

            print("Rank " + str(args.rank) + " GPUs:" + str(args.gpulist))
    else:    
        print('Not using distributed mode, using all available GPUs.')
        args.gpulist = list("cuda:" + str(p) for p in range(ngpus_per_node))
        return

    torch.cuda.set_device(args.gpu)
    #args.dist_backend = 'nccl'
    args.dist_backend = 'gloo'

    print('| distributed init (rank {}/{}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Init done, now calling barrier")
    torch.distributed.monitored_barrier()
    print("Barrier done, moving to setup")
    setup_for_distributed(args.rank == 0)
    print("Completed Distributed Setup")