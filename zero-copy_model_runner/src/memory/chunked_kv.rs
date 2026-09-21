// SPDX-License-Identifier: Apache-2.0
//! Chunked KV Cache Block Table and Allocator for Zero-Copy DMA-BUF Memory.
//!
//! Organizes KV cache storage into fixed-size physical memory blocks (chunks)
//! aligned to 64-byte hardware cache lines and page boundaries. This eliminates
//! virtual memory fragmentation during long-context generation and aligns directly
//! with Quest query-aware page bounding vectors and TriForce hierarchical retrieval.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ChunkedKvError {
    #[error("Out of physical KV chunk blocks: requested {requested}, available {available}")]
    OutOfBlocks { requested: usize, available: usize },
    #[error("Invalid sequence ID: {0}")]
    InvalidSequence(u32),
    #[error("Invalid chunk size {0}: must be greater than 0 and power of two")]
    InvalidChunkSize(usize),
    #[error("Block index {0} out of bounds (total blocks: {1})")]
    BlockOutOfBounds(usize, usize),
}

/// A physical chunk block representing storage for `chunk_size` tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkBlock {
    pub block_id: usize,
    pub chunk_size: usize,
    pub num_tokens: usize,
    pub byte_offset: usize,
    pub is_free: bool,
}

impl ChunkBlock {
    pub fn new(block_id: usize, chunk_size: usize, byte_offset: usize) -> Self {
        Self {
            block_id,
            chunk_size,
            num_tokens: 0,
            byte_offset,
            is_free: true,
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.chunk_size
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.chunk_size.saturating_sub(self.num_tokens)
    }
}

/// Configuration parameters for Chunked KV allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkedKvConfig {
    /// Tokens per chunk block (default: 32).
    pub chunk_size: usize,
    /// Total number of physical blocks pre-allocated in the DMA-BUF KV pool.
    pub total_blocks: usize,
    /// Element size per token in bytes (KV combined across all layers and heads).
    pub token_byte_stride: usize,
}

impl ChunkedKvConfig {
    /// Adjust token byte stride based on active Key-Value cache quantization.
    pub fn with_quant_type(mut self, quant_type: crate::memory::kv_quant::KvCacheQuantType, base_fp16_stride: usize) -> Self {
        let ratio = quant_type.effective_bytes_per_element() / 2.0;
        let quantized_stride = ((base_fp16_stride as f32 * ratio).ceil() as usize).max(16);
        // Align to 16-byte boundary for SIMD load efficiency
        self.token_byte_stride = (quantized_stride + 15) & !15;
        self
    }
}

impl Default for ChunkedKvConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32,
            total_blocks: 1024,
            token_byte_stride: 128, // Reference stride
        }
    }
}

/// Logical Block Table mapping sequences to physical DMA-BUF chunk blocks.
#[derive(Debug, Clone, Default)]
pub struct LogicalBlockTable {
    /// Mapping of sequence_id -> list of allocated physical block IDs.
    table: HashMap<u32, Vec<usize>>,
}

impl LogicalBlockTable {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn get_blocks(&self, seq_id: u32) -> Option<&[usize]> {
        self.table.get(&seq_id).map(|v| v.as_slice())
    }

    pub fn add_block(&mut self, seq_id: u32, block_id: usize) {
        self.table.entry(seq_id).or_default().push(block_id);
    }

    pub fn remove_sequence(&mut self, seq_id: u32) -> Option<Vec<usize>> {
        self.table.remove(&seq_id)
    }

    pub fn total_tokens(&self, seq_id: u32, blocks: &[ChunkBlock]) -> usize {
        if let Some(block_ids) = self.get_blocks(seq_id) {
            block_ids.iter().filter_map(|&id| blocks.get(id)).map(|b| b.num_tokens).sum()
        } else {
            0
        }
    }
}

/// Thread-safe physical chunk block allocator for zero-copy DMA-BUF memory.
#[derive(Debug)]
pub struct ChunkBlockAllocator {
    config: ChunkedKvConfig,
    blocks: Vec<ChunkBlock>,
    free_list: VecDeque<usize>,
    logical_table: LogicalBlockTable,
}

impl ChunkBlockAllocator {
    /// Initialize a new ChunkBlockAllocator with the given configuration.
    pub fn new(config: ChunkedKvConfig) -> Result<Self, ChunkedKvError> {
        if config.chunk_size == 0 || (config.chunk_size & (config.chunk_size - 1)) != 0 {
            return Err(ChunkedKvError::InvalidChunkSize(config.chunk_size));
        }

        let block_byte_size = config.chunk_size * config.token_byte_stride;
        let mut blocks = Vec::with_capacity(config.total_blocks);
        let mut free_list = VecDeque::with_capacity(config.total_blocks);

        for i in 0..config.total_blocks {
            let byte_offset = i * block_byte_size;
            blocks.push(ChunkBlock::new(i, config.chunk_size, byte_offset));
            free_list.push_back(i);
        }

        Ok(Self {
            config,
            blocks,
            free_list,
            logical_table: LogicalBlockTable::new(),
        })
    }

    /// Access configuration.
    pub fn config(&self) -> &ChunkedKvConfig {
        &self.config
    }

    /// Number of free physical blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of allocated physical blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.config.total_blocks - self.free_list.len()
    }

    /// Allocate $N$ tokens for a given sequence, allocating new blocks from the pool as needed.
    pub fn append_tokens(&mut self, seq_id: u32, num_tokens: usize) -> Result<Vec<usize>, ChunkedKvError> {
        let mut remaining = num_tokens;
        let mut touched_blocks = Vec::new();

        // 1. Fill current tail block of sequence if partially full
        if let Some(existing_blocks) = self.logical_table.get_blocks(seq_id) {
            if let Some(&last_block_id) = existing_blocks.last() {
                let last_block = &mut self.blocks[last_block_id];
                if !last_block.is_full() {
                    let fill = remaining.min(last_block.remaining_capacity());
                    last_block.num_tokens += fill;
                    remaining -= fill;
                    touched_blocks.push(last_block_id);
                }
            }
        }

        // 2. Allocate new blocks for remaining tokens
        let needed_blocks = (remaining + self.config.chunk_size - 1) / self.config.chunk_size;
        if needed_blocks > self.free_list.len() {
            return Err(ChunkedKvError::OutOfBlocks {
                requested: needed_blocks,
                available: self.free_list.len(),
            });
        }

        while remaining > 0 {
            let block_id = self.free_list.pop_front().unwrap();
            let block = &mut self.blocks[block_id];
            block.is_free = false;
            let fill = remaining.min(self.config.chunk_size);
            block.num_tokens = fill;
            remaining -= fill;

            self.logical_table.add_block(seq_id, block_id);
            touched_blocks.push(block_id);
        }

        Ok(touched_blocks)
    }

    /// Free all blocks associated with a sequence.
    pub fn free_sequence(&mut self, seq_id: u32) -> Result<usize, ChunkedKvError> {
        if let Some(block_ids) = self.logical_table.remove_sequence(seq_id) {
            let count = block_ids.len();
            for block_id in block_ids {
                let block = &mut self.blocks[block_id];
                block.is_free = true;
                block.num_tokens = 0;
                self.free_list.push_back(block_id);
            }
            Ok(count)
        } else {
            Ok(0)
        }
    }

    /// Return physical block IDs for active pages selected by Quest / TriForce.
    pub fn get_active_chunk_offsets(&self, seq_id: u32, active_page_indices: &[usize]) -> Vec<usize> {
        if let Some(blocks) = self.logical_table.get_blocks(seq_id) {
            active_page_indices
                .iter()
                .filter_map(|&p_idx| blocks.get(p_idx))
                .filter_map(|&block_id| self.blocks.get(block_id))
                .map(|b| b.byte_offset)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Return total active tokens in a sequence.
    pub fn sequence_tokens(&self, seq_id: u32) -> usize {
        self.logical_table.total_tokens(seq_id, &self.blocks)
    }

    /// Access physical blocks.
    pub fn blocks(&self) -> &[ChunkBlock] {
        &self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_kv_allocation_lifecycle() {
        let config = ChunkedKvConfig {
            chunk_size: 16,
            total_blocks: 8,
            token_byte_stride: 64,
        };
        let mut allocator = ChunkBlockAllocator::new(config).unwrap();

        assert_eq!(allocator.num_free_blocks(), 8);
        assert_eq!(allocator.num_allocated_blocks(), 0);

        // Allocate 35 tokens for sequence 0: requires ceil(35/16) = 3 blocks (16, 16, 3)
        let touched = allocator.append_tokens(0, 35).unwrap();
        assert_eq!(touched.len(), 3);
        assert_eq!(allocator.num_free_blocks(), 5);
        assert_eq!(allocator.num_allocated_blocks(), 3);
        assert_eq!(allocator.sequence_tokens(0), 35);

        // Append 10 more tokens to sequence 0: fills tail block (13) + 0 new blocks
        let touched2 = allocator.append_tokens(0, 10).unwrap();
        assert_eq!(touched2.len(), 1);
        assert_eq!(allocator.sequence_tokens(0), 45);
        assert_eq!(allocator.num_free_blocks(), 5);

        // Free sequence 0
        let freed = allocator.free_sequence(0).unwrap();
        assert_eq!(freed, 3);
        assert_eq!(allocator.num_free_blocks(), 8);
        assert_eq!(allocator.num_allocated_blocks(), 0);
    }

    #[test]
    fn test_chunked_kv_out_of_blocks() {
        let config = ChunkedKvConfig {
            chunk_size: 16,
            total_blocks: 2,
            token_byte_stride: 64,
        };
        let mut allocator = ChunkBlockAllocator::new(config).unwrap();

        // 32 tokens fill 2 blocks
        allocator.append_tokens(0, 32).unwrap();
        assert_eq!(allocator.num_free_blocks(), 0);

        // Requesting more fails cleanly with OutOfBlocks
        let err = allocator.append_tokens(0, 1).unwrap_err();
        assert!(matches!(err, ChunkedKvError::OutOfBlocks { .. }));
    }

    #[test]
    fn test_active_chunk_offsets() {
        let config = ChunkedKvConfig {
            chunk_size: 16,
            total_blocks: 4,
            token_byte_stride: 64,
        };
        let mut allocator = ChunkBlockAllocator::new(config).unwrap();

        allocator.append_tokens(0, 48).unwrap(); // 3 blocks (offsets 0, 1024, 2048)
        let offsets = allocator.get_active_chunk_offsets(0, &[0, 2]);
        assert_eq!(offsets, vec![0, 2048]);
    }
}
