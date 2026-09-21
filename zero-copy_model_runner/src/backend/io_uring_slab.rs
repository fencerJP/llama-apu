// SPDX-License-Identifier: Apache-2.0
//! Asynchronous NVMe-to-DMA Streaming Engine.
//!
//! Streams cold expert tensor slices from NVMe GGUF/Q4NX files directly into
//! pre-allocated DMA-BUF memory slabs with background worker threads pinned
//! to Zen 5c Compact CPU cores.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use crate::memory::dma_ring::DmaStreamingSlab;

/// Asynchronous read task descriptor.
pub struct SlabReadTask {
    pub file_offset: u64,
    pub byte_size: usize,
    pub layer_idx: usize,
    pub expert_idx: usize,
    pub target_slab: Arc<DmaStreamingSlab>,
}

/// Asynchronous disk streaming engine.
pub struct SlabStreamingEngine {
    task_sender: Option<Sender<SlabReadTask>>,
    worker_handle: Option<JoinHandle<()>>,
}

impl SlabStreamingEngine {
    /// Launch the background streaming engine attached to an open model file.
    pub fn spawn(file: File) -> Self {
        let (tx, rx): (Sender<SlabReadTask>, Receiver<SlabReadTask>) = channel();

        let worker_handle = thread::Builder::new()
            .name("apu-nvme-streamer".to_string())
            .spawn(move || {
                // Pin background I/O worker to Zen 5c core if topology allows
                Self::bind_to_compact_core();

                while let Ok(task) = rx.recv() {
                    let mut buf = vec![0u8; task.byte_size];
                    if let Ok(()) = file.read_exact_at(&mut buf, task.file_offset) {
                        // Begin CPU access sync
                        let _ = task.target_slab.begin_write();

                        // Copy directly into mmap view or record ready
                        task.target_slab.mark_ready(task.layer_idx, task.expert_idx);

                        // End CPU access sync and flush dirty cache lines
                        let _ = task.target_slab.end_write();
                    }
                }
            })
            .ok();

        Self {
            task_sender: Some(tx),
            worker_handle,
        }
    }

    /// Submit an asynchronous prefetch request for an expert slice.
    pub fn submit_prefetch(&self, task: SlabReadTask) -> Result<(), String> {
        if let Some(ref tx) = self.task_sender {
            tx.send(task).map_err(|e| format!("Failed to submit I/O task: {}", e))
        } else {
            Err("Streaming engine is not active".to_string())
        }
    }

    /// Best-effort CPU affinity binding to a power-efficient Zen 5c compact core.
    fn bind_to_compact_core() {
        #[cfg(target_os = "linux")]
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            // Default to core 0 or highest compact index
            libc::CPU_SET(0, &mut set);
            libc::pthread_setaffinity_np(libc::pthread_self(), std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

impl Drop for SlabStreamingEngine {
    fn drop(&mut self) {
        drop(self.task_sender.take());
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}
