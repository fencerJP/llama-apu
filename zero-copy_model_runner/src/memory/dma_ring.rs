// SPDX-License-Identifier: Apache-2.0
//! Pre-Allocated Circular DMA-BUF Ring Pool for Cold Expert Streaming.
//!
//! Provides a persistent ring of pre-allocated `dma-buf` memory slabs imported into
//! accelerator drivers (XRT / amdxdna) at startup, avoiding the ~11ms runtime IOMMU
//! page table pinning penalty.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use crate::memory::{DmaBufHandle, MemoryError, SyncDirection};

/// A pre-allocated DMA-BUF memory slab in the streaming ring.
#[derive(Debug)]
pub struct DmaStreamingSlab {
    /// Slab identifier index (0 .. slab_count - 1).
    pub slab_id: usize,
    /// Underlying Linux `dma-buf` buffer handle.
    pub dmabuf: DmaBufHandle,
    /// Capacity of this slab in bytes.
    pub capacity_bytes: usize,
    /// Whether this slab currently holds active, unconsumed data.
    in_use: AtomicBool,
    /// Currently loaded layer and expert index (if assigned).
    pub current_layer: AtomicUsize,
    pub current_expert: AtomicUsize,
}

impl DmaStreamingSlab {
    /// Create a new streaming slab from an existing DMA-BUF handle.
    pub fn new(slab_id: usize, dmabuf: DmaBufHandle) -> Self {
        let capacity_bytes = dmabuf.size();
        Self {
            slab_id,
            dmabuf,
            capacity_bytes,
            in_use: AtomicBool::new(false),
            current_layer: AtomicUsize::new(usize::MAX),
            current_expert: AtomicUsize::new(usize::MAX),
        }
    }

    /// Explicitly begin an asynchronous write session into this slab.
    pub fn begin_write(&self) -> Result<(), MemoryError> {
        self.dmabuf.begin_cpu_access(SyncDirection::Write)
    }

    /// End an asynchronous write session and flush dirty CPU write-combine lines before NPU read.
    pub fn end_write(&self) -> Result<(), MemoryError> {
        self.dmabuf.end_cpu_access(SyncDirection::Write)
    }

    /// Mark slab as containing ready data for an expert.
    pub fn mark_ready(&self, layer: usize, expert: usize) {
        self.current_layer.store(layer, Ordering::Release);
        self.current_expert.store(expert, Ordering::Release);
        self.in_use.store(true, Ordering::Release);
    }

    /// Release slab back to the ring pool.
    pub fn release(&self) {
        self.current_layer.store(usize::MAX, Ordering::Release);
        self.current_expert.store(usize::MAX, Ordering::Release);
        self.in_use.store(false, Ordering::Release);
    }

    /// Check if slab is available for streaming.
    pub fn is_free(&self) -> bool {
        !self.in_use.load(Ordering::Acquire)
    }
}

/// Circular pool of pre-allocated DMA-BUF streaming slabs.
#[derive(Debug)]
pub struct DmaStreamingRing {
    slabs: Vec<Arc<DmaStreamingSlab>>,
    head_index: AtomicUsize,
}

impl DmaStreamingRing {
    /// Construct a new streaming ring from a collection of pre-allocated DMA-BUF handles.
    pub fn new(handles: Vec<DmaBufHandle>) -> Self {
        let slabs = handles
            .into_iter()
            .enumerate()
            .map(|(idx, h)| Arc::new(DmaStreamingSlab::new(idx, h)))
            .collect();

        Self {
            slabs,
            head_index: AtomicUsize::new(0),
        }
    }

    /// Total number of slabs in the ring.
    #[inline]
    pub fn len(&self) -> usize {
        self.slabs.len()
    }

    /// Check if ring is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    /// Acquire the next available slab in circular sequence for an upcoming expert fetch.
    pub fn acquire_next_slab(&self) -> Option<Arc<DmaStreamingSlab>> {
        let total = self.slabs.len();
        if total == 0 {
            return None;
        }

        let start = self.head_index.fetch_add(1, Ordering::Relaxed) % total;

        for i in 0..total {
            let idx = (start + i) % total;
            let slab = &self.slabs[idx];
            if slab.is_free() {
                return Some(slab.clone());
            }
        }

        // If all slabs are currently marked in-use, return the oldest head slab (ring rollover)
        Some(self.slabs[start].clone())
    }

    /// Access slab by index.
    pub fn get_slab(&self, index: usize) -> Option<Arc<DmaStreamingSlab>> {
        self.slabs.get(index).cloned()
    }
}
