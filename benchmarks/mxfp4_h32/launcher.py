"""Build a standard, explicit multi-node torchrun command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaunchGeometry:
    nodes: int
    processes_per_node: int
    node_rank: int
    master_addr: str
    master_port: int
    rendezvous_id: str

    def validate(self) -> None:
        if self.nodes < 1 or self.processes_per_node < 1:
            raise ValueError("nodes and processes_per_node must be positive")
        if self.node_rank not in range(self.nodes):
            raise ValueError("node_rank is outside the launch topology")
        if not self.master_addr or "://" in self.master_addr:
            raise ValueError("master_addr must be a hostname or address, not a URI")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port is outside 1..65535")
        if not self.rendezvous_id or any(
            character.isspace() for character in self.rendezvous_id
        ):
            raise ValueError("rendezvous_id must be a non-empty token")


def torchrun_command(
    geometry: LaunchGeometry,
    train_entry: Path,
    train_arguments: list[str],
) -> list[str]:
    geometry.validate()
    return [
        "torchrun",
        "--nnodes",
        str(geometry.nodes),
        "--nproc_per_node",
        str(geometry.processes_per_node),
        "--node_rank",
        str(geometry.node_rank),
        "--rdzv_endpoint",
        f"{geometry.master_addr}:{geometry.master_port}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_id",
        geometry.rendezvous_id,
        "--rdzv-conf",
        "timeout=3600,read_timeout=3600",
        "--max_restarts",
        "0",
        str(train_entry),
        *train_arguments,
    ]
