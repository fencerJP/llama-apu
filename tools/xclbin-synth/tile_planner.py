#!/usr/bin/env python3
"""Milestone 7.1.1: Spatial Tile Allocator & Memory Planner for AMD Ryzen AI NPU.

Plans 4x8 (AIE2P) or 4x5 (AIE2) spatial tile arrays, verifies 16-byte Tile DMA alignment,
and calculates on-chip SRAM allocations for MoE router matrices and ping-pong ObjectFIFOs.
"""

from dataclasses import asdict, dataclass
import sys
from typing import Dict, List, Optional
from model_topology import ModelTopology, extract_topology


@dataclass
class HardwareTarget:
    name: str
    rows: int
    cols: int
    total_tiles: int
    tile_sram_bytes: int
    total_sram_bytes: int
    dma_alignment_bytes: int = 16


TARGET_NPU1_AIE2 = HardwareTarget(
    name="npu1-aie2",
    rows=4,
    cols=5,
    total_tiles=20,
    tile_sram_bytes=32 * 1024,      # 32 KiB local SRAM
    total_sram_bytes=512 * 1024,    # 512 KiB shared array SRAM
    dma_alignment_bytes=16,
)

TARGET_NPU2_AIE2P = HardwareTarget(
    name="npu2-aie2p",
    rows=4,
    cols=8,
    total_tiles=32,
    tile_sram_bytes=32 * 1024,      # 32 KiB local SRAM
    total_sram_bytes=4 * 1024 * 1024, # 4 MiB aggregate array SRAM
    dma_alignment_bytes=16,
)


@dataclass
class RouterSramPlan:
    total_router_bytes: int
    pinned_sram_bytes: int
    pinned_layer_count: int
    total_layers: int
    sram_exhaustion_prevented: bool


@dataclass
class SpatialTilePlan:
    target_hardware: str
    array_shape: List[int]
    compute_tiles: int
    memory_tiles: int
    tile_scratchpad_bytes: int
    object_fifo_buffer_bytes: int
    dma_alignment_verified: bool
    unaligned_tensors: List[str]
    router_sram_plan: RouterSramPlan


def plan_spatial_tiles(
    topo: ModelTopology,
    target: HardwareTarget = TARGET_NPU2_AIE2P,
    router_sram_enabled: bool = True,
    max_router_sram_mb: int = 32,
) -> SpatialTilePlan:
    """Compute optimal spatial placement and memory partitions."""
    cols = target.cols
    rows = target.rows
    total = target.total_tiles

    # On AIE2P: top row (cols tiles) or edge columns allocated as memory/DMA tiles
    mem_tiles = cols
    comp_tiles = total - mem_tiles

    # Ping-pong double buffering fits in 32 KiB tile SRAM:
    # Buffer A (14 KiB) + Buffer B (14 KiB) + 4 KiB stack/scratch = 32 KiB
    fifo_buf_bytes = 14 * 1024
    scratch_bytes = 32 * 1024

    # Verify 16-byte Tile DMA alignment across key tensor dimensions
    unaligned = []
    tensors_to_check = {
        "hidden_dim": topo.hidden_dim,
        "ffn_dim": topo.ffn_dim,
        "head_dim": topo.head_dim,
    }
    # For T-ACE 16B co-packed representation, 64 weights map to 16 bytes (1/4 byte per trit payload + scale)
    # Check if dimensions align to 64 weights (16 bytes)
    for name, val in tensors_to_check.items():
        if (val % 16) != 0:
            unaligned.append(f"{name} ({val}) not 16-byte divisible")

    # MoE Router SRAM planning
    max_sram_bytes = max_router_sram_mb * 1024 * 1024
    if router_sram_enabled and topo.num_experts > 0 and topo.num_layers > 0:
        # W_gate shape [hidden_dim, num_experts] in FP16 (2 bytes)
        bytes_per_layer = topo.hidden_dim * topo.num_experts * 2
        total_router_bytes = bytes_per_layer * topo.num_layers
        capped_bytes = min(total_router_bytes, max_sram_bytes)
        pinned_layers = min(capped_bytes // bytes_per_layer, topo.num_layers) if bytes_per_layer > 0 else 0
        pinned_bytes = pinned_layers * bytes_per_layer
        sram_prevented = total_router_bytes > max_sram_bytes
    else:
        total_router_bytes = 0
        pinned_bytes = 0
        pinned_layers = 0
        sram_prevented = False

    rplan = RouterSramPlan(
        total_router_bytes=total_router_bytes,
        pinned_sram_bytes=pinned_bytes,
        pinned_layer_count=pinned_layers,
        total_layers=topo.num_layers,
        sram_exhaustion_prevented=sram_prevented,
    )

    return SpatialTilePlan(
        target_hardware=target.name,
        array_shape=[rows, cols],
        compute_tiles=comp_tiles,
        memory_tiles=mem_tiles,
        tile_scratchpad_bytes=scratch_bytes,
        object_fifo_buffer_bytes=fifo_buf_bytes,
        dma_alignment_verified=(len(unaligned) == 0),
        unaligned_tensors=unaligned,
        router_sram_plan=rplan,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tile_planner.py <model.gguf | config.json>")
        sys.exit(1)
    topo = extract_topology(sys.argv[1])
    plan = plan_spatial_tiles(topo)
    import json
    print(json.dumps(asdict(plan), indent=2))
