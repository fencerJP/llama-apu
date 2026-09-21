// SPDX-License-Identifier: Apache-2.0
use std::ffi::CString;
use std::os::fd::{AsRawFd, IntoRawFd};
use nix::sys::memfd::{memfd_create, MemFdCreateFlag};
use zero_copy_model_runner::memory::{DmaBufHandle, DmaStreamingRing};

#[test]
fn test_dma_streaming_ring_lifecycle() {
    let slab_size = 1024 * 1024; // 1 MB per slab
    let num_slabs = 4;

    let mut handles = Vec::new();
    for i in 0..num_slabs {
        let name = CString::new(format!("test_slab_{}", i)).unwrap();
        let owned_fd = memfd_create(&name, MemFdCreateFlag::MFD_CLOEXEC)
            .expect("memfd_create should succeed");
        nix::unistd::ftruncate(&owned_fd, slab_size as i64).unwrap();
        let raw_fd = owned_fd.into_raw_fd();
        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(raw_fd, slab_size) };
        handles.push(handle);
    }

    let ring = DmaStreamingRing::new(handles);
    assert_eq!(ring.len(), 4);
    assert!(!ring.is_empty());

    // Acquire next available slab
    let slab_0 = ring.acquire_next_slab().expect("Slab 0 should be acquired");
    assert_eq!(slab_0.slab_id, 0);
    assert!(slab_0.is_free());

    // Begin CPU write sync
    assert!(slab_0.begin_write().is_ok());

    // Mark ready with layer 0, expert 5
    slab_0.mark_ready(0, 5);
    assert!(!slab_0.is_free());

    // End CPU write sync
    assert!(slab_0.end_write().is_ok());

    // Acquire next slab (should get slab 1)
    let slab_1 = ring.acquire_next_slab().expect("Slab 1 should be acquired");
    assert_eq!(slab_1.slab_id, 1);

    // Release slab 0
    slab_0.release();
    assert!(slab_0.is_free());
}
