import logging
import os
import socket
from datetime import timedelta
from typing import Any, Optional, Union
from urllib.parse import urlparse

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    ProcessGroupNCCL,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


def stateless_init_process_group(init_method, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.

    Args:
        init_method: TCP init method string (e.g., "tcp://localhost:9000")
        rank: The rank of this process in the group
        world_size: Total number of processes in the group
        device: The CUDA device to use for NCCL communication
    """
    # Parse master_address and master_port from init_method (e.g., "tcp://localhost:9000")
    parsed = urlparse(init_method)
    master_address = parsed.hostname or "localhost"
    master_port = parsed.port or 9000
    logger.info(
        "Initializing StatelessProcessGroup: init_method=%s host=%s port=%s rank=%s world_size=%s device=%s",
        init_method,
        master_address,
        master_port,
        rank,
        world_size,
        device,
    )
    logger.info(
        "Env: MASTER_ADDR=%s MASTER_PORT=%s RANK=%s WORLD_SIZE=%s HOSTNAME=%s FQDN=%s",
        os.environ.get("MASTER_ADDR"),
        os.environ.get("MASTER_PORT"),
        os.environ.get("RANK"),
        os.environ.get("WORLD_SIZE"),
        socket.gethostname(),
        socket.getfqdn(),
    )
    resolved_addrs = set()
    try:
        for info in socket.getaddrinfo(master_address, None):
            resolved_addrs.add(info[4][0])
    except OSError as exc:
        logger.warning("Failed to resolve master_address=%s: %s", master_address, exc)
    local_addrs = set()
    try:
        import psutil  # type: ignore

        iface_addrs = {}
        for if_name, addrs in psutil.net_if_addrs().items():
            iface_addrs[if_name] = []
            for addr in addrs:
                if addr.family in (socket.AF_INET, socket.AF_INET6):
                    local_addrs.add(addr.address)
                    iface_addrs[if_name].append(addr.address)
        if iface_addrs:
            logger.info("Interface addresses: %s", iface_addrs)
    except Exception as exc:
        try:
            local_addrs.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        logger.debug("Unable to read full local interface list: %s", exc)
    if resolved_addrs:
        logger.info("Resolved master_address=%s to %s", master_address, sorted(resolved_addrs))
    if local_addrs:
        logger.info("Local IPs: %s", sorted(local_addrs))
    if resolved_addrs and local_addrs and not (resolved_addrs & local_addrs):
        logger.warning(
            "master_address does not appear to be local on this node. "
            "If you see Errno 99 (Cannot assign requested address), the bind host is likely wrong."
        )

    try:
        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
    except OSError as exc:
        logger.error(
            "Failed to create StatelessProcessGroup with host=%s port=%s: %s",
            master_address,
            master_port,
            exc,
        )
        raise
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_extra_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # Create NCCL-specific options if using NCCL backend
    logger.info(f"[{group_name}] Backend: {backend}, str(backend): {str(backend)}")
    if pg_options is None and str(backend) == "nccl":
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False
        logger.info(f"[{group_name}] Created NCCL options: {pg_options}")

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=pg_options,
        timeout=timeout,
    )
    logger.info(f"[{group_name}] Process group created successfully")

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
