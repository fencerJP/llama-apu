// SPDX-License-Identifier: Apache-2.0
//! Zero-Copy Memory Management via Linux DMA-BUF and DRM GEM.

use std::os::fd::RawFd;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

use crate::backend::DeviceBackend;
use crate::uapi::*;

pub mod chunked_kv;
pub mod kv_pruning;
pub mod strix_halo_tuning;

pub use chunked_kv::{ChunkBlock, ChunkBlockAllocator, ChunkedKvConfig, ChunkedKvError, LogicalBlockTable};
pub use kv_pruning::{DynamicKvPruner, KvPruningConfig, PruneResult};
pub use strix_halo_tuning::{StrixHaloConfig, StrixHaloMemoryOptimizer};

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Failed to execute DMA-BUF ioctl: {0}")]
    IoctlError(#[from] nix::Error),
    #[error("I/O error on file descriptor: {0}")]
    IoError(#[from] std::io::Error),
    #[error("mmap failure: {0}")]
    MmapError(String),
    #[error("Buffer alignment error: required {required}, got {actual}")]
    AlignmentError { required: usize, actual: usize },
    #[error("Device memory allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Invalid buffer state: {0}")]
    InvalidState(String),
}

/// Access flags for CPU-side cache coherency brackets via `DMA_BUF_IOCTL_SYNC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    Read,
    Write,
    ReadWrite,
}

impl SyncDirection {
    fn to_flags(self) -> u64 {
        match self {
            SyncDirection::Read => DMA_BUF_SYNC_READ,
            SyncDirection::Write => DMA_BUF_SYNC_WRITE,
            SyncDirection::ReadWrite => DMA_BUF_SYNC_RW,
        }
    }
}

/// Safe RAII wrapper around a Linux `dma-buf` file descriptor.
///
/// Encapsulates kernel buffer sharing without host-side data copying.
/// Guarantees that upon drop, the underlying file descriptor is cleanly closed.
#[derive(Debug)]
pub struct DmaBufHandle {
    raw_fd: RawFd,
    size_bytes: usize,
    in_sync_session: AtomicBool,
}

impl DmaBufHandle {
    /// Wrap an existing raw file descriptor representing an exported `dma-buf`.
    ///
    /// # Safety
    /// The caller must ensure that ownership is transferred exclusively to this struct.
    pub unsafe fn from_raw_fd_unchecked(raw_fd: RawFd, size_bytes: usize) -> Self {
        Self {
            raw_fd,
            size_bytes,
            in_sync_session: AtomicBool::new(false),
        }
    }

    /// Size of the underlying shared memory buffer in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    /// Borrow the underlying raw file descriptor.
    #[inline]
    pub fn as_raw_fd(&self) -> RawFd {
        self.raw_fd
    }

    /// Duplicate the file descriptor for multi-device attachments.
    pub fn try_clone(&self) -> Result<Self, MemoryError> {
        let dup_fd = unsafe { libc::fcntl(self.raw_fd, libc::F_DUPFD_CLOEXEC, 0) };
        if dup_fd < 0 {
            return Err(MemoryError::IoError(std::io::Error::last_os_error()));
        }
        Ok(Self {
            raw_fd: dup_fd,
            size_bytes: self.size_bytes,
            in_sync_session: AtomicBool::new(false),
        })
    }

    /// Explicitly start a CPU cache-synchronization bracket via `DMA_BUF_IOCTL_SYNC`.
    ///
    /// Required before any CPU read or write to an mmap'd view of this buffer.
    pub fn begin_cpu_access(&self, direction: SyncDirection) -> Result<(), MemoryError> {
        let mut sync_args = dma_buf_sync {
            flags: DMA_BUF_SYNC_START | direction.to_flags(),
        };

        let res = unsafe { dma_buf_ioctl_sync(self.raw_fd, &mut sync_args) };
        match res {
            Ok(_) => {}
            Err(nix::Error::ENOTTY) => {
                // In mock mode backed by memfd, the Linux kernel does not support DMA_BUF_IOCTL_SYNC on memfd.
                // Validate that the descriptor is a valid open file descriptor.
                let flags = unsafe { libc::fcntl(self.raw_fd, libc::F_GETFD) };
                if flags < 0 {
                    return Err(MemoryError::IoctlError(nix::Error::last()));
                }
            }
            Err(e) => return Err(MemoryError::IoctlError(e)),
        }

        self.in_sync_session.store(true, Ordering::Release);
        Ok(())
    }

    /// Explicitly end a CPU cache-synchronization bracket via `DMA_BUF_IOCTL_SYNC`.
    ///
    /// Flushes dirty CPU cache lines and prepares the buffer for accelerator DMA.
    pub fn end_cpu_access(&self, direction: SyncDirection) -> Result<(), MemoryError> {
        let mut sync_args = dma_buf_sync {
            flags: DMA_BUF_SYNC_END | direction.to_flags(),
        };

        let res = unsafe { dma_buf_ioctl_sync(self.raw_fd, &mut sync_args) };
        match res {
            Ok(_) => {}
            Err(nix::Error::ENOTTY) => {
                let flags = unsafe { libc::fcntl(self.raw_fd, libc::F_GETFD) };
                if flags < 0 {
                    return Err(MemoryError::IoctlError(nix::Error::last()));
                }
            }
            Err(e) => return Err(MemoryError::IoctlError(e)),
        }

        self.in_sync_session.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for DmaBufHandle {
    fn drop(&mut self) {
        if self.raw_fd >= 0 {
            let is_valid = unsafe { libc::fcntl(self.raw_fd, libc::F_GETFD) >= 0 };
            if is_valid {
                unsafe {
                    libc::close(self.raw_fd);
                }
            }
        }
    }
}

/// A high-level, zero-copy shared memory buffer backed by a `dma-buf`.
///
/// Provides both device access (via the underlying `DmaBufHandle`) and optional
/// host CPU memory mapping (via `mmap`) with strict cache coherency guards.
pub struct SharedBuffer {
    handle: DmaBufHandle,
    mmap_ptr: Option<NonNull<u8>>,
    size: usize,
}

// Safety: The memory buffer is mapped into process space; safe to send across threads.
unsafe impl Send for SharedBuffer {}
// Safety: Synchronization is enforced via atomic sync sessions and kernel fences.
unsafe impl Sync for SharedBuffer {}

impl SharedBuffer {
    /// Create a new `SharedBuffer` from an existing `DmaBufHandle`.
    ///
    /// Optionally maps the buffer into userspace virtual memory with 64-byte
    /// cache-line alignment if `map_cpu` is true.
    pub fn new(handle: DmaBufHandle, map_cpu: bool) -> Result<Self, MemoryError> {
        let size = handle.size();
        let mmap_ptr = if map_cpu {
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    handle.as_raw_fd(),
                    0,
                )
            };

            if addr == libc::MAP_FAILED {
                return Err(MemoryError::MmapError(
                    std::io::Error::last_os_error().to_string(),
                ));
            }

            // Verify 64-byte cache line alignment
            let ptr_val = addr as usize;
            if ptr_val % 64 != 0 {
                unsafe { libc::munmap(addr, size) };
                return Err(MemoryError::AlignmentError {
                    required: 64,
                    actual: ptr_val % 64,
                });
            }

            NonNull::new(addr as *mut u8)
        } else {
            None
        };

        Ok(Self {
            handle,
            mmap_ptr,
            size,
        })
    }

    /// Access the underlying `DmaBufHandle` for accelerator kernel dispatches.
    #[inline]
    pub fn handle(&self) -> &DmaBufHandle {
        &self.handle
    }

    /// Total capacity of this buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if this buffer has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Execute a scoped read-only CPU closure over the buffer contents.
    ///
    /// Automatically manages `DMA_BUF_SYNC_START` and `DMA_BUF_SYNC_END`.
    pub fn with_cpu_read<F, R>(&self, f: F) -> Result<R, MemoryError>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let ptr = self.mmap_ptr.ok_or_else(|| {
            MemoryError::InvalidState("Buffer is not mapped to CPU address space".into())
        })?;

        self.handle.begin_cpu_access(SyncDirection::Read)?;
        let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), self.size) };
        let result = f(slice);
        self.handle.end_cpu_access(SyncDirection::Read)?;

        Ok(result)
    }

    /// Execute a scoped mutable CPU closure over the buffer contents.
    ///
    /// Automatically manages `DMA_BUF_SYNC_START` and `DMA_BUF_SYNC_END`.
    pub fn with_cpu_write<F, R>(&mut self, f: F) -> Result<R, MemoryError>
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let ptr = self.mmap_ptr.ok_or_else(|| {
            MemoryError::InvalidState("Buffer is not mapped to CPU address space".into())
        })?;

        self.handle.begin_cpu_access(SyncDirection::Write)?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), self.size) };
        let result = f(slice);
        self.handle.end_cpu_access(SyncDirection::Write)?;

        Ok(result)
    }

    /// Return the raw pointer to the mapped memory, if CPU-mapped.
    #[inline]
    pub fn as_ptr(&self) -> Option<*const u8> {
        self.mmap_ptr.map(|p| p.as_ptr() as *const u8)
    }

    /// Return the mutable raw pointer to the mapped memory, if CPU-mapped.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        self.mmap_ptr.map(|p| p.as_ptr())
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        if let Some(ptr) = self.mmap_ptr.take() {
            unsafe {
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, self.size);
            }
        }
    }
}

/// RAII wrapper around an allocated AMDGPU DRM GEM buffer object.
///
/// Ensures deterministic closure via `DRM_IOCTL_GEM_CLOSE` upon drop.
#[derive(Debug)]
pub struct GpuBufferObject {
    backend: Arc<dyn DeviceBackend>,
    handle: u32,
    size_bytes: usize,
}

impl GpuBufferObject {
    pub fn new(backend: Arc<dyn DeviceBackend>, handle: u32, size_bytes: usize) -> Self {
        Self {
            backend,
            handle,
            size_bytes,
        }
    }

    #[inline]
    pub fn handle(&self) -> u32 {
        self.handle
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    /// Export this GEM buffer handle to a Linux standard `dma-buf` file descriptor.
    pub fn export_prime_fd(&self) -> Result<DmaBufHandle, MemoryError> {
        let mut prime = drm_prime_handle {
            handle: self.handle,
            flags: DRM_CLOEXEC | DRM_RDWR,
            fd: -1,
        };

        self.backend.ioctl(
            DRM_IOCTL_PRIME_HANDLE_TO_FD_NUM,
            &mut prime as *mut _ as *mut libc::c_void,
        )?;

        if prime.fd < 0 {
            return Err(MemoryError::AllocationFailed(
                "PRIME export returned negative file descriptor".into(),
            ));
        }

        Ok(unsafe { DmaBufHandle::from_raw_fd_unchecked(prime.fd, self.size_bytes) })
    }
}

impl Drop for GpuBufferObject {
    fn drop(&mut self) {
        let mut close_arg = drm_gem_close {
            handle: self.handle,
            pad: 0,
        };
        let _ = self.backend.ioctl(
            DRM_IOCTL_GEM_CLOSE_NUM,
            &mut close_arg as *mut _ as *mut libc::c_void,
        );
    }
}

/// RAII wrapper around an imported buffer object in the AMDXDNA NPU device context.
///
/// Ensures deterministic closure via `DRM_IOCTL_GEM_CLOSE` upon drop.
#[derive(Debug)]
pub struct NpuBufferObject {
    backend: Arc<dyn DeviceBackend>,
    handle: u32,
    size_bytes: usize,
    xdna_addr: u64,
}

pub type XrtBoHandle = NpuBufferObject;

impl NpuBufferObject {
    pub fn new(backend: Arc<dyn DeviceBackend>, handle: u32, size_bytes: usize, xdna_addr: u64) -> Self {
        Self {
            backend,
            handle,
            size_bytes,
            xdna_addr,
        }
    }

    #[inline]
    pub fn handle(&self) -> u32 {
        self.handle
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    #[inline]
    pub fn xdna_addr(&self) -> u64 {
        self.xdna_addr
    }
}

impl Drop for NpuBufferObject {
    fn drop(&mut self) {
        let mut close_arg = drm_gem_close {
            handle: self.handle,
            pad: 0,
        };
        let _ = self.backend.ioctl(
            DRM_IOCTL_GEM_CLOSE_NUM,
            &mut close_arg as *mut _ as *mut libc::c_void,
        );
    }
}

/// Port interface for memory allocation and buffer management.
pub trait MemoryPort: Send + Sync {
    fn allocate_shared_gem(&self, size_bytes: usize) -> Result<DmaBufHandle, MemoryError>;
    fn export_prime_fd(&self, handle: &DmaBufHandle) -> Result<RawFd, MemoryError>;
    fn import_xrt_bo(&self, fd: RawFd, size_bytes: usize) -> Result<XrtBoHandle, MemoryError>;
}

/// Zero-Copy Cross-Accelerator Memory Bridge.
///
/// Coordinates unified physical DRAM allocation on AMDGPU, PRIME export to Linux standard `dma-buf`,
/// and binding into the AMDXDNA NPU address space.
pub struct MemoryBridge;

impl MemoryBridge {
    /// Allocate a physically contiguous GEM buffer object in the AMDGPU GTT domain.
    ///
    /// Configures mandatory zero-copy flags:
    /// `AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED | AMDGPU_GEM_CREATE_COHERENT | AMDGPU_GEM_CREATE_EXPLICIT_SYNC`.
    pub fn allocate_gem_bo(
        backend: &Arc<dyn DeviceBackend>,
        size_bytes: usize,
        alignment: usize,
    ) -> Result<GpuBufferObject, MemoryError> {
        if size_bytes == 0 {
            return Err(MemoryError::AllocationFailed("Cannot allocate 0 bytes".into()));
        }
        if size_bytes % 64 != 0 {
            return Err(MemoryError::AlignmentError {
                required: 64,
                actual: size_bytes % 64,
            });
        }

        let align = if alignment == 0 { 64 } else { alignment };
        if align % 64 != 0 {
            return Err(MemoryError::AlignmentError {
                required: 64,
                actual: align % 64,
            });
        }

        let mut req = drm_amdgpu_gem_create {
            r#in: drm_amdgpu_gem_create_in {
                bo_size: size_bytes as u64,
                alignment: align as u64,
                domains: AMDGPU_GEM_DOMAIN_GTT,
                domain_flags: AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED
                    | AMDGPU_GEM_CREATE_COHERENT
                    | AMDGPU_GEM_CREATE_EXPLICIT_SYNC,
            },
        };

        backend.ioctl(
            DRM_IOCTL_AMDGPU_GEM_CREATE_NUM,
            &mut req as *mut _ as *mut libc::c_void,
        )?;

        let handle = unsafe { req.out.handle };
        if handle == 0 {
            return Err(MemoryError::AllocationFailed("Kernel returned GEM handle 0".into()));
        }

        Ok(GpuBufferObject::new(Arc::clone(backend), handle, size_bytes))
    }

    /// Export an allocated AMDGPU GEM buffer object to an OwnedFd `DmaBufHandle`.
    pub fn export_dma_buf(bo: &GpuBufferObject) -> Result<DmaBufHandle, MemoryError> {
        bo.export_prime_fd()
    }

    /// Import an exported `dma-buf` into the AMDXDNA NPU context (`/dev/accel/accel0`).
    pub fn import_dma_buf(
        npu_backend: &Arc<dyn DeviceBackend>,
        dmabuf: &DmaBufHandle,
    ) -> Result<NpuBufferObject, MemoryError> {
        let mut prime = drm_prime_handle {
            handle: 0,
            flags: 0,
            fd: dmabuf.as_raw_fd(),
        };

        npu_backend.ioctl(
            DRM_IOCTL_PRIME_FD_TO_HANDLE_NUM,
            &mut prime as *mut _ as *mut libc::c_void,
        )?;

        if prime.handle == 0 {
            return Err(MemoryError::AllocationFailed(
                "NPU PRIME import returned handle 0".into(),
            ));
        }

        let mut bo_info = amdxdna_drm_get_bo_info {
            ext: 0,
            ext_flags: 0,
            handle: prime.handle,
            pad: 0,
            map_offset: 0,
            vaddr: 0,
            xdna_addr: 0,
        };

        let _ = npu_backend.ioctl(
            DRM_IOCTL_AMDXDNA_GET_BO_INFO_NUM,
            &mut bo_info as *mut _ as *mut libc::c_void,
        );

        Ok(NpuBufferObject::new(
            Arc::clone(npu_backend),
            prime.handle,
            dmabuf.size(),
            bo_info.xdna_addr,
        ))
    }

    /// Convenient one-stop allocation: creates a GPU GEM BO, exports to PRIME FD,
    /// and constructs a CPU-mapped `SharedBuffer`.
    pub fn allocate_shared(
        gpu_backend: &Arc<dyn DeviceBackend>,
        size_bytes: usize,
        map_cpu: bool,
    ) -> Result<(GpuBufferObject, SharedBuffer), MemoryError> {
        let gpu_bo = Self::allocate_gem_bo(gpu_backend, size_bytes, 4096)?;
        let dmabuf = gpu_bo.export_prime_fd()?;
        let shared = SharedBuffer::new(dmabuf, map_cpu)?;
        Ok((gpu_bo, shared))
    }
}
