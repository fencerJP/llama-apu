// SPDX-License-Identifier: Apache-2.0
//! In-memory mock accelerator device backend for rootless testing and CI environments.
//!
//! Simulates AMDGPU and AMDXDNA driver behaviors using Linux `memfd_create` and
//! atomic timeline fence state machines. Validates kernel struct layouts, 64-byte
//! cache-line alignments, and fence transitions without requiring physical silicon.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use super::{DeviceBackend, DeviceType};
use crate::uapi::*;

/// Global registry tracking exported PRIME dma-buf file descriptors and buffer sizes.
static GLOBAL_DMABUF_REGISTRY: OnceLock<Mutex<HashMap<RawFd, usize>>> = OnceLock::new();

fn dmabuf_registry() -> &'static Mutex<HashMap<RawFd, usize>> {
    GLOBAL_DMABUF_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Global registry tracking exported syncobj file descriptors and timeline states.
static GLOBAL_SYNCOBJ_REGISTRY: OnceLock<Mutex<HashMap<RawFd, Arc<MockSyncobjState>>>> = OnceLock::new();

fn syncobj_registry() -> &'static Mutex<HashMap<RawFd, Arc<MockSyncobjState>>> {
    GLOBAL_SYNCOBJ_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Simulated timeline fence state.
#[derive(Debug)]
pub struct MockSyncobjState {
    pub timeline: AtomicU64,
    pub condvar: Condvar,
    pub mutex: Mutex<()>,
}

impl MockSyncobjState {
    pub fn new(initial_point: u64) -> Self {
        Self {
            timeline: AtomicU64::new(initial_point),
            condvar: Condvar::new(),
            mutex: Mutex::new(()),
        }
    }

    pub fn current_point(&self) -> u64 {
        self.timeline.load(Ordering::Acquire)
    }

    pub fn signal(&self, point: u64) {
        let _guard = self.mutex.lock().unwrap();
        let prev = self.timeline.fetch_max(point, Ordering::SeqCst);
        if point > prev {
            self.condvar.notify_all();
        }
    }

    pub fn wait(&self, point: u64, timeout_nsec: i64) -> Result<(), nix::Error> {
        if self.timeline.load(Ordering::Acquire) >= point {
            return Ok(());
        }

        let mut guard = self.mutex.lock().unwrap();
        if self.timeline.load(Ordering::Acquire) >= point {
            return Ok(());
        }

        if timeout_nsec == 0 {
            return Err(nix::Error::ETIMEDOUT);
        }

        if timeout_nsec < 0 {
            // Infinite wait
            while self.timeline.load(Ordering::Acquire) < point {
                guard = self.condvar.wait(guard).unwrap();
            }
            Ok(())
        } else {
            let timeout = Duration::from_nanos(timeout_nsec as u64);
            let start = std::time::Instant::now();
            while self.timeline.load(Ordering::Acquire) < point {
                let elapsed = start.elapsed();
                if elapsed >= timeout {
                    return Err(nix::Error::ETIMEDOUT);
                }
                let remaining = timeout - elapsed;
                let (new_guard, timeout_result) = self.condvar.wait_timeout(guard, remaining).unwrap();
                guard = new_guard;
                if timeout_result.timed_out() && self.timeline.load(Ordering::Acquire) < point {
                    return Err(nix::Error::ETIMEDOUT);
                }
            }
            Ok(())
        }
    }
}

/// Representation of an allocated mock GEM buffer backed by an in-memory `memfd`.
#[allow(dead_code)]
#[derive(Debug)]
struct MockGemBuffer {
    memfd: OwnedFd,
    size: usize,
    alignment: usize,
    domains: u64,
    domain_flags: u64,
}

/// High-fidelity mock accelerator backend.
#[derive(Debug)]
pub struct MockDeviceBackend {
    dev_type: DeviceType,
    dummy_fd: OwnedFd,
    next_gem_handle: AtomicU32,
    next_syncobj_handle: AtomicU32,
    gem_buffers: Mutex<HashMap<u32, MockGemBuffer>>,
    syncobjs: Mutex<HashMap<u32, Arc<MockSyncobjState>>>,
}

impl MockDeviceBackend {
    /// Instantiate a mock accelerator backend of specified type.
    pub fn new(dev_type: DeviceType) -> Self {
        let name = match dev_type {
            DeviceType::Gpu => b"mock_amdgpu\0".as_ptr() as *const libc::c_char,
            DeviceType::Npu => b"mock_amdxdna\0".as_ptr() as *const libc::c_char,
        };

        let raw = unsafe { libc::memfd_create(name, libc::MFD_CLOEXEC) };
        assert!(raw >= 0, "Failed to create mock device dummy memfd");
        let dummy_fd = unsafe { OwnedFd::from_raw_fd(raw) };

        Self {
            dev_type,
            dummy_fd,
            next_gem_handle: AtomicU32::new(1),
            next_syncobj_handle: AtomicU32::new(1),
            gem_buffers: Mutex::new(HashMap::new()),
            syncobjs: Mutex::new(HashMap::new()),
        }
    }

    /// Convenience constructor for mock GPU.
    pub fn new_gpu() -> Self {
        Self::new(DeviceType::Gpu)
    }

    /// Convenience constructor for mock NPU.
    pub fn new_npu() -> Self {
        Self::new(DeviceType::Npu)
    }
}

impl DeviceBackend for MockDeviceBackend {
    fn ioctl(&self, request: u64, arg: *mut libc::c_void) -> Result<i32, nix::Error> {
        if arg.is_null() {
            return Err(nix::Error::EFAULT);
        }

        match request {
            // -----------------------------------------------------------------
            // AMDGPU GEM Allocation
            // -----------------------------------------------------------------
            DRM_IOCTL_AMDGPU_GEM_CREATE_NUM => {
                let req = unsafe { &mut *(arg as *mut drm_amdgpu_gem_create) };
                let in_args = unsafe { req.r#in };

                // Validate buffer size and strict 64-byte alignment
                if in_args.bo_size == 0 {
                    return Err(nix::Error::EINVAL);
                }
                if in_args.bo_size % 64 != 0 {
                    return Err(nix::Error::EINVAL);
                }
                if in_args.alignment != 0 && in_args.alignment % 64 != 0 {
                    return Err(nix::Error::EINVAL);
                }
                // Validate domain includes GTT for cross-device APU zero-copy sharing
                if (in_args.domains & AMDGPU_GEM_DOMAIN_GTT) == 0 {
                    return Err(nix::Error::EINVAL);
                }

                // Allocate backing physical RAM via memfd_create
                let memfd_name = b"mock_amdgpu_gem_bo\0".as_ptr() as *const libc::c_char;
                let raw_memfd = unsafe { libc::memfd_create(memfd_name, libc::MFD_CLOEXEC) };
                if raw_memfd < 0 {
                    return Err(nix::Error::last());
                }

                let ftruncate_res = unsafe { libc::ftruncate(raw_memfd, in_args.bo_size as libc::off_t) };
                if ftruncate_res < 0 {
                    let err = nix::Error::last();
                    unsafe { libc::close(raw_memfd) };
                    return Err(err);
                }

                let handle = self.next_gem_handle.fetch_add(1, Ordering::SeqCst);
                let bo = MockGemBuffer {
                    memfd: unsafe { OwnedFd::from_raw_fd(raw_memfd) },
                    size: in_args.bo_size as usize,
                    alignment: in_args.alignment as usize,
                    domains: in_args.domains,
                    domain_flags: in_args.domain_flags,
                };

                self.gem_buffers.lock().unwrap().insert(handle, bo);

                req.out = drm_amdgpu_gem_create_out {
                    handle,
                    _pad: 0,
                };

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM GEM Close
            // -----------------------------------------------------------------
            DRM_IOCTL_GEM_CLOSE_NUM => {
                let close_arg = unsafe { &*(arg as *const drm_gem_close) };
                let mut buffers = self.gem_buffers.lock().unwrap();
                buffers.remove(&close_arg.handle);
                Ok(0)
            }

            // -----------------------------------------------------------------
            // PRIME Export (Handle -> FD)
            // -----------------------------------------------------------------
            DRM_IOCTL_PRIME_HANDLE_TO_FD_NUM => {
                let prime_arg = unsafe { &mut *(arg as *mut drm_prime_handle) };
                let buffers = self.gem_buffers.lock().unwrap();
                let bo = buffers.get(&prime_arg.handle).ok_or(nix::Error::ENOENT)?;

                // Duplicate the memfd to yield an independent descriptor sharing identical physical pages
                let dup_fd = unsafe { libc::fcntl(bo.memfd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
                if dup_fd < 0 {
                    return Err(nix::Error::last());
                }

                // Register exported dma-buf in global registry for cross-driver import
                dmabuf_registry().lock().unwrap().insert(dup_fd, bo.size);
                prime_arg.fd = dup_fd;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // PRIME Import (FD -> Handle)
            // -----------------------------------------------------------------
            DRM_IOCTL_PRIME_FD_TO_HANDLE_NUM => {
                let prime_arg = unsafe { &mut *(arg as *mut drm_prime_handle) };
                let fd = prime_arg.fd;

                // Validate descriptor
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                if flags < 0 {
                    return Err(nix::Error::EBADF);
                }

                let size = {
                    let reg = dmabuf_registry().lock().unwrap();
                    if let Some(&s) = reg.get(&fd) {
                        s
                    } else {
                        let cur_len = unsafe { libc::lseek(fd, 0, libc::SEEK_END) };
                        if cur_len < 0 {
                            return Err(nix::Error::last());
                        }
                        cur_len as usize
                    }
                };

                let dup_fd = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
                if dup_fd < 0 {
                    return Err(nix::Error::last());
                }

                let handle = self.next_gem_handle.fetch_add(1, Ordering::SeqCst);
                let bo = MockGemBuffer {
                    memfd: unsafe { OwnedFd::from_raw_fd(dup_fd) },
                    size,
                    alignment: 64,
                    domains: AMDGPU_GEM_DOMAIN_GTT,
                    domain_flags: AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED | AMDGPU_GEM_CREATE_COHERENT,
                };

                self.gem_buffers.lock().unwrap().insert(handle, bo);
                prime_arg.handle = handle;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // AMDXDNA Create BO
            // -----------------------------------------------------------------
            DRM_IOCTL_AMDXDNA_CREATE_BO_NUM => {
                let bo_arg = unsafe { &mut *(arg as *mut amdxdna_drm_create_bo) };

                // Validate: When dmabuf_fd is used in va_tbl, num_entries must be zero
                if bo_arg.vaddr != 0 {
                    let va_tbl = unsafe { &*(bo_arg.vaddr as *const amdxdna_drm_va_tbl) };
                    if va_tbl.dmabuf_fd != 0 && va_tbl.num_entries != 0 {
                        return Err(nix::Error::EINVAL);
                    }
                }

                let handle = self.next_gem_handle.fetch_add(1, Ordering::SeqCst);
                bo_arg.handle = handle;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // AMDXDNA Get BO Info
            // -----------------------------------------------------------------
            DRM_IOCTL_AMDXDNA_GET_BO_INFO_NUM => {
                let info_arg = unsafe { &mut *(arg as *mut amdxdna_drm_get_bo_info) };
                if info_arg.handle == 0 {
                    return Err(nix::Error::EINVAL);
                }

                info_arg.map_offset = 0x100000000 + (info_arg.handle as u64) * 0x100000;
                info_arg.vaddr = 0;
                info_arg.xdna_addr = 0x200000000 + (info_arg.handle as u64) * 0x100000;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Create
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_CREATE_NUM => {
                let create_arg = unsafe { &mut *(arg as *mut drm_syncobj_create) };
                let initial_point = if (create_arg.flags & 1) != 0 { 1 } else { 0 };
                let handle = self.next_syncobj_handle.fetch_add(1, Ordering::SeqCst);
                let state = Arc::new(MockSyncobjState::new(initial_point));

                self.syncobjs.lock().unwrap().insert(handle, state);
                create_arg.handle = handle;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Destroy
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_DESTROY_NUM => {
                let destroy_arg = unsafe { &*(arg as *const drm_syncobj_destroy) };
                self.syncobjs.lock().unwrap().remove(&destroy_arg.handle);
                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Handle to FD
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM => {
                let h2fd = unsafe { &mut *(arg as *mut drm_syncobj_handle) };
                let syncobjs = self.syncobjs.lock().unwrap();
                let state = syncobjs.get(&h2fd.handle).ok_or(nix::Error::ENOENT)?;

                let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
                if efd < 0 {
                    return Err(nix::Error::last());
                }

                syncobj_registry().lock().unwrap().insert(efd, Arc::clone(state));
                h2fd.fd = efd;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj FD to Handle
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_NUM => {
                let fd2h = unsafe { &mut *(arg as *mut drm_syncobj_handle) };
                let reg = syncobj_registry().lock().unwrap();
                let state = reg.get(&fd2h.fd).ok_or(nix::Error::ENOENT)?;

                let handle = self.next_syncobj_handle.fetch_add(1, Ordering::SeqCst);
                self.syncobjs.lock().unwrap().insert(handle, Arc::clone(state));
                fd2h.handle = handle;

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Timeline Signal
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM => {
                let arr = unsafe { &*(arg as *const drm_syncobj_timeline_array) };
                let handles_ptr = arr.handles as *const u32;
                let points_ptr = arr.points as *const u64;

                let syncobjs = self.syncobjs.lock().unwrap();
                for i in 0..arr.count_handles {
                    let handle = unsafe { *handles_ptr.add(i as usize) };
                    let point = unsafe { *points_ptr.add(i as usize) };
                    if let Some(state) = syncobjs.get(&handle) {
                        state.signal(point);
                    } else {
                        return Err(nix::Error::ENOENT);
                    }
                }

                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Timeline Wait
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM => {
                let wait_arg = unsafe { &mut *(arg as *mut drm_syncobj_timeline_wait) };
                let handles_ptr = wait_arg.handles as *const u32;
                let points_ptr = wait_arg.points as *const u64;

                let states: Vec<(Arc<MockSyncobjState>, u64)> = {
                    let syncobjs = self.syncobjs.lock().unwrap();
                    let mut list = Vec::with_capacity(wait_arg.count_handles as usize);
                    for i in 0..wait_arg.count_handles {
                        let handle = unsafe { *handles_ptr.add(i as usize) };
                        let point = unsafe { *points_ptr.add(i as usize) };
                        let state = syncobjs.get(&handle).ok_or(nix::Error::ENOENT)?;
                        list.push((Arc::clone(state), point));
                    }
                    list
                };

                for (state, target_point) in states {
                    state.wait(target_point, wait_arg.timeout_nsec)?;
                }

                wait_arg.first_signaled = 0;
                Ok(0)
            }

            // -----------------------------------------------------------------
            // DRM Syncobj Transfer
            // -----------------------------------------------------------------
            DRM_IOCTL_SYNCOBJ_TRANSFER_NUM => {
                let transfer = unsafe { &*(arg as *const drm_syncobj_transfer) };
                let syncobjs = self.syncobjs.lock().unwrap();
                let src_state = syncobjs.get(&transfer.src_handle).ok_or(nix::Error::ENOENT)?;
                let dst_state = syncobjs.get(&transfer.dst_handle).ok_or(nix::Error::ENOENT)?;

                let point = src_state.current_point();
                dst_state.signal(point);

                Ok(0)
            }

            // -----------------------------------------------------------------
            // Direct DMA_BUF_IOCTL_SYNC validation on backend
            // -----------------------------------------------------------------
            DMA_BUF_IOCTL_SYNC_NUM => {
                let sync_arg = unsafe { &*(arg as *const dma_buf_sync) };
                if (sync_arg.flags & !DMA_BUF_SYNC_VALID_FLAGS_MASK) != 0 {
                    return Err(nix::Error::EINVAL);
                }
                Ok(0)
            }

            _ => Err(nix::Error::ENOTTY),
        }
    }

    fn as_raw_fd(&self) -> RawFd {
        self.dummy_fd.as_raw_fd()
    }

    fn is_mock(&self) -> bool {
        true
    }

    fn device_type(&self) -> DeviceType {
        self.dev_type
    }
}
