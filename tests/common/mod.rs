// SPDX-License-Identifier: Apache-2.0
//! Common Test Fixtures, Silicon Probes, Mock Hardware Drivers, and Invariant Verifiers.
//!
//! Provides progressive testability across both physical AMD Ryzen AI APU silicon
//! (/dev/dri/renderD128, /dev/accel/accel0) and software mock emulation shims.

use std::fs::OpenOptions;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, DecodeStepResult, EngineError, PrefillEngine, PrefillRequest,
    PrefillResult, SpeculativeDraftingEngine,
};
use zero_copy_model_runner::memory::DmaBufHandle;

// ============================================================================
// 1. Authoritative Linux Kernel UAPI Definitions & Constants
// ============================================================================

pub const DRM_IOCTL_BASE: u8 = b'd';
pub const DRM_COMMAND_BASE: u8 = 0x40;

// AMDGPU GEM
pub const DRM_AMDGPU_GEM_CREATE: u8 = 0x00;
pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
pub const AMDGPU_GEM_CREATE_COHERENT: u64 = 1 << 13;
pub const AMDGPU_GEM_CREATE_EXPLICIT_SYNC: u64 = 1 << 7;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_amdgpu_gem_create_in {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_amdgpu_gem_create_out {
    pub handle: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union drm_amdgpu_gem_create {
    pub r#in: drm_amdgpu_gem_create_in,
    pub out: drm_amdgpu_gem_create_out,
}

// Compile-time size assertion: 32 bytes per kernel header
const _: () = assert!(std::mem::size_of::<drm_amdgpu_gem_create>() == 32);

// GEM Close
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_gem_close {
    pub handle: u32,
    pub pad: u32,
}
const _: () = assert!(std::mem::size_of::<drm_gem_close>() == 8);

// DRM PRIME
pub const DRM_CLOEXEC: u32 = 0x80000;
pub const DRM_RDWR: u32 = 0x00002;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_prime_handle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
}
const _: () = assert!(std::mem::size_of::<drm_prime_handle>() == 12);

// DRM Syncobj
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1 << 0;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE: u32 = 1 << 2;
pub const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE: u32 = 1 << 3;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_create {
    pub handle: u32,
    pub flags: u32,
}
const _: () = assert!(std::mem::size_of::<drm_syncobj_create>() == 8);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_destroy {
    pub handle: u32,
    pub pad: u32,
}
const _: () = assert!(std::mem::size_of::<drm_syncobj_destroy>() == 8);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_handle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
    pub pad: u32,
    pub point: u64,
}
const _: () = assert!(std::mem::size_of::<drm_syncobj_handle>() == 24);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_timeline_wait {
    pub handles: u64,
    pub points: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
    pub deadline_nsec: u64,
}
const _: () = assert!(std::mem::size_of::<drm_syncobj_timeline_wait>() == 48);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_timeline_array {
    pub handles: u64,
    pub points: u64,
    pub count_handles: u32,
    pub pad: u32,
}
const _: () = assert!(std::mem::size_of::<drm_syncobj_timeline_array>() == 24);

// Linux DMA-BUF
pub const DMA_BUF_SYNC_READ: u64 = 1 << 0;
pub const DMA_BUF_SYNC_WRITE: u64 = 2 << 0;
pub const DMA_BUF_SYNC_RW: u64 = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE;
pub const DMA_BUF_SYNC_START: u64 = 0 << 2;
pub const DMA_BUF_SYNC_END: u64 = 1 << 2;
pub const DMA_BUF_SYNC_VALID_FLAGS_MASK: u64 = DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct dma_buf_sync {
    pub flags: u64,
}
const _: () = assert!(std::mem::size_of::<dma_buf_sync>() == 8);

// IOCTL wrapper macros using nix
nix::ioctl_readwrite!(amdgpu_gem_create_ioctl, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE, drm_amdgpu_gem_create);
nix::ioctl_write_ptr!(drm_gem_close_ioctl, DRM_IOCTL_BASE, 0x09, drm_gem_close);
nix::ioctl_readwrite!(drm_prime_handle_to_fd_ioctl, DRM_IOCTL_BASE, 0x2D, drm_prime_handle);
nix::ioctl_readwrite!(drm_prime_fd_to_handle_ioctl, DRM_IOCTL_BASE, 0x2E, drm_prime_handle);
nix::ioctl_readwrite!(drm_syncobj_create_ioctl, DRM_IOCTL_BASE, 0xBF, drm_syncobj_create);
nix::ioctl_readwrite!(drm_syncobj_destroy_ioctl, DRM_IOCTL_BASE, 0xC0, drm_syncobj_destroy);
nix::ioctl_readwrite!(drm_syncobj_handle_to_fd_ioctl, DRM_IOCTL_BASE, 0xC1, drm_syncobj_handle);
nix::ioctl_readwrite!(drm_syncobj_fd_to_handle_ioctl, DRM_IOCTL_BASE, 0xC2, drm_syncobj_handle);
nix::ioctl_readwrite!(drm_syncobj_timeline_wait_ioctl, DRM_IOCTL_BASE, 0xCA, drm_syncobj_timeline_wait);
nix::ioctl_readwrite!(drm_syncobj_timeline_signal_ioctl, DRM_IOCTL_BASE, 0xCD, drm_syncobj_timeline_array);
nix::ioctl_write_ptr!(dma_buf_ioctl_sync, b'b', 0, dma_buf_sync);

// ============================================================================
// 2. Physical Silicon Probe & Detection
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct SiliconStatus {
    pub has_amdgpu: bool,
    pub has_amdxdna: bool,
    pub has_full_silicon: bool,
}

pub struct SiliconProbe;

impl SiliconProbe {
    pub const RENDER_NODE: &'static str = "/dev/dri/renderD128";
    pub const ACCEL_NODE: &'static str = "/dev/accel/accel0";

    pub fn probe() -> SiliconStatus {
        let has_amdgpu = OpenOptions::new()
            .read(true)
            .write(true)
            .open(Self::RENDER_NODE)
            .is_ok();

        let has_amdxdna = OpenOptions::new()
            .read(true)
            .write(true)
            .open(Self::ACCEL_NODE)
            .is_ok();

        SiliconStatus {
            has_amdgpu,
            has_amdxdna,
            has_full_silicon: has_amdgpu && has_amdxdna,
        }
    }

    pub fn open_render_node() -> Result<OwnedFd, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(Self::RENDER_NODE)
            .map_err(|e| format!("Failed to open {}: {}", Self::RENDER_NODE, e))?;
        Ok(file.into())
    }

    pub fn open_accel_node() -> Result<OwnedFd, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(Self::ACCEL_NODE)
            .map_err(|e| format!("Failed to open {}: {}", Self::ACCEL_NODE, e))?;
        Ok(file.into())
    }
}

// ============================================================================
// 3. Physical Silicon Hardware Bridge
// ============================================================================

pub struct PhysicalHardwareBridge;

impl PhysicalHardwareBridge {
    pub fn allocate_and_export_gem(
        render_fd: RawFd,
        size_bytes: usize,
    ) -> Result<(u32, OwnedFd), String> {
        let aligned_size = (size_bytes + 4095) & !4095;
        let mut gem_create = drm_amdgpu_gem_create {
            r#in: drm_amdgpu_gem_create_in {
                bo_size: aligned_size as u64,
                alignment: 4096,
                domains: AMDGPU_GEM_DOMAIN_GTT,
                domain_flags: AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED
                    | AMDGPU_GEM_CREATE_COHERENT
                    | AMDGPU_GEM_CREATE_EXPLICIT_SYNC,
            },
        };

        unsafe {
            amdgpu_gem_create_ioctl(render_fd, &mut gem_create)
                .map_err(|e| format!("AMDGPU GEM create failed: {}", e))?;
        }

        let gem_handle = unsafe { gem_create.out.handle };

        let mut prime = drm_prime_handle {
            handle: gem_handle,
            flags: DRM_CLOEXEC | DRM_RDWR,
            fd: -1,
        };

        unsafe {
            drm_prime_handle_to_fd_ioctl(render_fd, &mut prime)
                .map_err(|e| format!("PRIME handle to FD failed: {}", e))?;
        }

        if prime.fd < 0 {
            return Err("Negative FD returned from PRIME export".into());
        }

        let owned_dmabuf_fd = unsafe { OwnedFd::from_raw_fd(prime.fd) };
        Ok((gem_handle, owned_dmabuf_fd))
    }

    pub fn import_dmabuf_to_npu(accel_fd: RawFd, dmabuf_fd: RawFd) -> Result<u32, String> {
        let mut prime = drm_prime_handle {
            handle: 0,
            flags: 0,
            fd: dmabuf_fd,
        };

        unsafe {
            drm_prime_fd_to_handle_ioctl(accel_fd, &mut prime)
                .map_err(|e| format!("PRIME FD to handle on accel0 failed: {}", e))?;
        }

        Ok(prime.handle)
    }

    pub fn close_gem_handle(device_fd: RawFd, handle: u32) -> Result<(), String> {
        let close_args = drm_gem_close { handle, pad: 0 };
        unsafe {
            drm_gem_close_ioctl(device_fd, &close_args)
                .map_err(|e| format!("GEM close failed for handle {}: {}", handle, e))?;
        }
        Ok(())
    }

    pub fn create_syncobj(device_fd: RawFd) -> Result<u32, String> {
        let mut create = drm_syncobj_create { handle: 0, flags: 0 };
        unsafe {
            drm_syncobj_create_ioctl(device_fd, &mut create)
                .map_err(|e| format!("Syncobj create failed: {}", e))?;
        }
        Ok(create.handle)
    }

    pub fn destroy_syncobj(device_fd: RawFd, handle: u32) -> Result<(), String> {
        let mut destroy = drm_syncobj_destroy { handle, pad: 0 };
        unsafe {
            drm_syncobj_destroy_ioctl(device_fd, &mut destroy)
                .map_err(|e| format!("Syncobj destroy failed: {}", e))?;
        }
        Ok(())
    }

    pub fn export_syncobj_to_fd(device_fd: RawFd, handle: u32) -> Result<OwnedFd, String> {
        let mut h2fd = drm_syncobj_handle {
            handle,
            flags: 0,
            fd: -1,
            pad: 0,
            point: 0,
        };
        unsafe {
            drm_syncobj_handle_to_fd_ioctl(device_fd, &mut h2fd)
                .map_err(|e| format!("Syncobj handle to FD failed: {}", e))?;
        }
        if h2fd.fd < 0 {
            return Err("Negative FD returned from syncobj export".into());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(h2fd.fd) })
    }

    pub fn import_syncobj_from_fd(device_fd: RawFd, sync_fd: RawFd) -> Result<u32, String> {
        let mut fd2h = drm_syncobj_handle {
            handle: 0,
            flags: 0,
            fd: sync_fd,
            pad: 0,
            point: 0,
        };
        unsafe {
            drm_syncobj_fd_to_handle_ioctl(device_fd, &mut fd2h)
                .map_err(|e| format!("Syncobj FD to handle failed: {}", e))?;
        }
        Ok(fd2h.handle)
    }

    pub fn timeline_signal(device_fd: RawFd, handle: u32, point: u64) -> Result<(), String> {
        let handles = [handle];
        let points = [point];
        let mut timeline_array = drm_syncobj_timeline_array {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            count_handles: 1,
            pad: 0,
        };
        unsafe {
            drm_syncobj_timeline_signal_ioctl(device_fd, &mut timeline_array)
                .map_err(|e| format!("Timeline signal failed for point {}: {}", point, e))?;
        }
        Ok(())
    }

    pub fn timeline_wait(
        device_fd: RawFd,
        handle: u32,
        point: u64,
        timeout_ms: u64,
        flags: u32,
    ) -> Result<(), String> {
        let handles = [handle];
        let points = [point];
        let now_ns = unsafe {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
            (ts.tv_sec as i64 * 1_000_000_000) + ts.tv_nsec as i64
        };
        let timeout_nsec = now_ns + (timeout_ms as i64 * 1_000_000);

        let mut wait = drm_syncobj_timeline_wait {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            timeout_nsec,
            count_handles: 1,
            flags,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };

        unsafe {
            drm_syncobj_timeline_wait_ioctl(device_fd, &mut wait)
                .map_err(|e| format!("Timeline wait failed on point {}: {}", point, e))?;
        }
        Ok(())
    }
}

// ============================================================================
// 4. Mock Hardware Harness & Emulation Driver (memfd-backed)
// ============================================================================

#[derive(Debug, Default)]
pub struct MockTimelineSyncobj {
    current_point: AtomicU64,
}

impl MockTimelineSyncobj {
    pub fn new(initial_point: u64) -> Self {
        Self {
            current_point: AtomicU64::new(initial_point),
        }
    }

    pub fn signal(&self, point: u64) -> Result<(), String> {
        let mut curr = self.current_point.load(Ordering::Acquire);
        loop {
            if point < curr {
                return Err(format!(
                    "Non-monotonic signal point: requested {}, current {}",
                    point, curr
                ));
            }
            match self.current_point.compare_exchange_weak(
                curr,
                point,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => curr = actual,
            }
        }
    }

    pub fn wait(&self, point: u64, timeout: Duration) -> Result<(), String> {
        let start = Instant::now();
        while self.current_point.load(Ordering::Acquire) < point {
            if start.elapsed() >= timeout {
                return Err(format!(
                    "Mock timeline fence timeout waiting for point {}",
                    point
                ));
            }
            std::thread::yield_now();
        }
        Ok(())
    }

    pub fn current_point(&self) -> u64 {
        self.current_point.load(Ordering::Acquire)
    }
}

pub struct MockDeviceHarness {
    timeline_fences: Mutex<std::collections::HashMap<u32, Arc<MockTimelineSyncobj>>>,
    next_handle: Mutex<u32>,
}

impl MockDeviceHarness {
    pub fn new() -> Self {
        Self {
            timeline_fences: Mutex::new(std::collections::HashMap::new()),
            next_handle: Mutex::new(1),
        }
    }

    pub fn create_mock_dmabuf(&self, size_bytes: usize) -> Result<OwnedFd, String> {
        let aligned_size = (size_bytes + 4095) & !4095;
        let cname = std::ffi::CString::new("mock_dma_buf_kv_cache").map_err(|e| e.to_string())?;
        let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(format!("memfd_create failed: {}", std::io::Error::last_os_error()));
        }

        if unsafe { libc::ftruncate(fd, aligned_size as libc::off_t) } != 0 {
            unsafe { libc::close(fd) };
            return Err(format!("ftruncate failed: {}", std::io::Error::last_os_error()));
        }

        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    pub fn create_syncobj(&self) -> u32 {
        let mut handle_guard = self.next_handle.lock().unwrap();
        let handle = *handle_guard;
        *handle_guard += 1;

        let fence = Arc::new(MockTimelineSyncobj::new(0));
        self.timeline_fences.lock().unwrap().insert(handle, fence);
        handle
    }

    pub fn destroy_syncobj(&self, handle: u32) -> Result<(), String> {
        let mut fences = self.timeline_fences.lock().unwrap();
        if fences.remove(&handle).is_some() {
            Ok(())
        } else {
            Err(format!("Invalid mock syncobj handle {}", handle))
        }
    }

    pub fn timeline_signal(&self, handle: u32, point: u64) -> Result<(), String> {
        let fences = self.timeline_fences.lock().unwrap();
        let fence = fences.get(&handle).ok_or_else(|| format!("Handle {} not found", handle))?;
        fence.signal(point)
    }

    pub fn timeline_wait(&self, handle: u32, point: u64, timeout: Duration) -> Result<(), String> {
        let fence = {
            let fences = self.timeline_fences.lock().unwrap();
            fences.get(&handle).cloned().ok_or_else(|| format!("Handle {} not found", handle))?
        };
        fence.wait(point, timeout)
    }

    pub fn validate_sync_flags(flags: u64) -> Result<(), String> {
        let dir = flags & DMA_BUF_SYNC_RW;
        if dir != DMA_BUF_SYNC_READ && dir != DMA_BUF_SYNC_WRITE && dir != DMA_BUF_SYNC_RW {
            return Err(format!("Invalid sync direction flags: 0x{:x}", flags));
        }
        if (flags & !DMA_BUF_SYNC_VALID_FLAGS_MASK) != 0 {
            return Err(format!("Extraneous invalid flags in dma_buf_sync: 0x{:x}", flags));
        }
        Ok(())
    }
}

// ============================================================================
// 5. High-Fidelity Mock Compute Engines (Prefill & Decode)
// ============================================================================

pub struct MockPrefillEngine {
    is_initialized: AtomicBool,
    device_index: Mutex<u32>,
    peak_compute_tflops: f32,
    total_tokens_processed: AtomicU64,
}

impl MockPrefillEngine {
    pub fn new() -> Self {
        Self {
            is_initialized: AtomicBool::new(false),
            device_index: Mutex::new(0),
            peak_compute_tflops: 32.0,
            total_tokens_processed: AtomicU64::new(0),
        }
    }

    pub fn total_tokens_processed(&self) -> u64 {
        self.total_tokens_processed.load(Ordering::Acquire)
    }
}

impl PrefillEngine for MockPrefillEngine {
    fn initialize(&mut self, device_index: u32) -> Result<(), EngineError> {
        *self.device_index.lock().unwrap() = device_index;
        self.is_initialized.store(true, Ordering::Release);
        Ok(())
    }

    fn dispatch_prefill(
        &self,
        request: PrefillRequest,
        kv_cache: &DmaBufHandle,
        syncobj_fd: RawFd,
        signal_timeline_point: u64,
    ) -> Result<PrefillResult, EngineError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(EngineError::InitFailed("PrefillEngine not initialized".into()));
        }

        if request.token_ids.is_empty() {
            return Err(EngineError::InvalidArgument("Prompt token sequence is empty".into()));
        }

        if request.token_ids.len() > 8192 {
            return Err(EngineError::InvalidArgument(format!(
                "Prompt length {} exceeds maximum supported 8192 tokens",
                request.token_ids.len()
            )));
        }

        for &token_id in request.token_ids {
            if token_id > 128_256 {
                return Err(EngineError::InvalidArgument(format!(
                    "Token ID {} exceeds vocabulary size 128,000",
                    token_id
                )));
            }
        }

        let num_tokens = request.token_ids.len();
        let bytes_per_token = 128;
        let required_bytes = (request.start_offset + num_tokens) * bytes_per_token;
        if kv_cache.size() < required_bytes {
            return Err(EngineError::OutOfResources(format!(
                "KV cache buffer capacity {} bytes is less than required {} bytes",
                kv_cache.size(),
                required_bytes
            )));
        }

        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                kv_cache.size(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                kv_cache.as_raw_fd(),
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(EngineError::OutOfResources("Failed to mmap KV cache for prefill".into()));
            }

            assert_eq!(
                ptr as usize % 64,
                0,
                "KV cache mmap pointer must be strictly 64-byte cache-line aligned"
            );

            let slice = std::slice::from_raw_parts_mut(
                (ptr as *mut u8).add(request.start_offset * bytes_per_token),
                num_tokens * bytes_per_token,
            );
            for (i, &token) in request.token_ids.iter().enumerate() {
                let token_slice = &mut slice[i * bytes_per_token..(i + 1) * bytes_per_token];
                token_slice.fill((token % 251) as u8);
            }

            libc::munmap(ptr, kv_cache.size());
        }

        self.total_tokens_processed
            .fetch_add(num_tokens as u64, Ordering::Release);

        if syncobj_fd >= 0 {
            let handles = [1u32];
            let points = [signal_timeline_point];
            let mut array = drm_syncobj_timeline_array {
                handles: handles.as_ptr() as u64,
                points: points.as_ptr() as u64,
                count_handles: 1,
                pad: 0,
            };
            let _ = unsafe { drm_syncobj_timeline_signal_ioctl(syncobj_fd, &mut array) };
        }

        let initial_token_id = DeterministicReferenceOracle::next_token(request.token_ids);

        Ok(PrefillResult {
            tokens_processed: num_tokens,
            initial_token_id,
            execution_time_us: (num_tokens as u64) * 85,
            completion_fence_point: signal_timeline_point,
        })
    }

    fn peak_compute_tflops(&self) -> f32 {
        self.peak_compute_tflops
    }
}

pub struct MockDecodeEngine {
    is_initialized: AtomicBool,
    xclbin_path: Mutex<String>,
    attached_kv_size: AtomicU64,
    attached_kv_raw_fd: Mutex<Option<RawFd>>,
    active_tiles: u32,
    peak_npu_tops: f32,
    current_step: AtomicU64,
}

impl MockDecodeEngine {
    pub fn new() -> Self {
        Self {
            is_initialized: AtomicBool::new(false),
            xclbin_path: Mutex::new(String::new()),
            attached_kv_size: AtomicU64::new(0),
            attached_kv_raw_fd: Mutex::new(None),
            active_tiles: 32,
            peak_npu_tops: 55.0,
            current_step: AtomicU64::new(0),
        }
    }

    pub fn current_step(&self) -> u64 {
        self.current_step.load(Ordering::Acquire)
    }
}

impl DecodeEngine for MockDecodeEngine {
    fn initialize(&mut self, xclbin_path: &str) -> Result<(), EngineError> {
        *self.xclbin_path.lock().unwrap() = xclbin_path.to_string();
        self.is_initialized.store(true, Ordering::Release);
        Ok(())
    }

    fn attach_kv_cache(&mut self, kv_cache: &DmaBufHandle) -> Result<(), EngineError> {
        if kv_cache.size() == 0 {
            return Err(EngineError::InvalidArgument("KV cache size cannot be 0".into()));
        }
        self.attached_kv_size.store(kv_cache.size() as u64, Ordering::Release);
        *self.attached_kv_raw_fd.lock().unwrap() = Some(kv_cache.as_raw_fd());
        Ok(())
    }

    fn dispatch_decode_step(
        &self,
        request: DecodeStepRequest,
        wait_syncobj_fd: RawFd,
        wait_timeline_point: u64,
        signal_timeline_point: u64,
    ) -> Result<DecodeStepResult, EngineError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(EngineError::InitFailed("DecodeEngine not initialized".into()));
        }

        let kv_raw_fd = self.attached_kv_raw_fd.lock().unwrap().ok_or_else(|| {
            EngineError::InvalidArgument("No KV cache attached to DecodeEngine".into())
        })?;

        if request.sequence_index >= 8192 {
            return Err(EngineError::InvalidArgument(format!(
                "Sequence index {} exceeds maximum context 8192",
                request.sequence_index
            )));
        }

        if request.temperature < 0.0 {
            return Err(EngineError::InvalidArgument(format!(
                "Temperature {} cannot be negative",
                request.temperature
            )));
        }

        if wait_syncobj_fd >= 0 && wait_timeline_point > 0 {
            let handles = [1u32];
            let points = [wait_timeline_point];
            let now_ns = unsafe {
                let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
                (ts.tv_sec as i64 * 1_000_000_000) + ts.tv_nsec as i64
            };
            let mut wait = drm_syncobj_timeline_wait {
                handles: handles.as_ptr() as u64,
                points: points.as_ptr() as u64,
                timeout_nsec: now_ns + 100_000_000,
                count_handles: 1,
                flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
                first_signaled: 0,
                pad: 0,
                deadline_nsec: 0,
            };
            let _ = unsafe { drm_syncobj_timeline_wait_ioctl(wait_syncobj_fd, &mut wait) };
        }

        let bytes_per_token = 128;
        let offset = request.sequence_index * bytes_per_token;
        let kv_size = self.attached_kv_size.load(Ordering::Acquire) as usize;
        if offset + bytes_per_token > kv_size {
            return Err(EngineError::OutOfResources(format!(
                "Sequence index {} exceeds allocated KV cache buffer capacity",
                request.sequence_index
            )));
        }

        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                kv_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                kv_raw_fd,
                0,
            );
            if ptr != libc::MAP_FAILED {
                assert_eq!(ptr as usize % 64, 0);
                let slice = std::slice::from_raw_parts_mut(
                    (ptr as *mut u8).add(offset),
                    bytes_per_token,
                );
                slice.fill((request.input_token_id % 251) as u8);
                libc::munmap(ptr, kv_size);
            }
        }

        self.current_step.fetch_add(1, Ordering::Release);

        if wait_syncobj_fd >= 0 && signal_timeline_point > 0 {
            let handles = [1u32];
            let points = [signal_timeline_point];
            let mut array = drm_syncobj_timeline_array {
                handles: handles.as_ptr() as u64,
                points: points.as_ptr() as u64,
                count_handles: 1,
                pad: 0,
            };
            let _ = unsafe { drm_syncobj_timeline_signal_ioctl(wait_syncobj_fd, &mut array) };
        }

        let output_token_id =
            DeterministicReferenceOracle::next_decode_step_token(request.input_token_id, request.sequence_index);
        let is_eos = output_token_id == DeterministicReferenceOracle::EOS_TOKEN_ID;

        Ok(DecodeStepResult {
            output_token_id,
            is_eos,
            tile_latency_us: 28_000,
            step_fence_point: signal_timeline_point,
        })
    }

    fn active_tiles(&self) -> u32 {
        self.active_tiles
    }

    fn peak_npu_tops(&self) -> f32 {
        self.peak_npu_tops
    }
}

pub struct MockSpeculativeDraftingEngine {
    _draft_length: usize,
}

impl MockSpeculativeDraftingEngine {
    pub fn new(draft_length: usize) -> Self {
        Self { _draft_length: draft_length }
    }
}

impl SpeculativeDraftingEngine for MockSpeculativeDraftingEngine {
    fn draft_tokens(
        &self,
        seed_token_id: u32,
        draft_length: usize,
    ) -> Result<Vec<u32>, EngineError> {
        if draft_length == 0 {
            return Err(EngineError::InvalidArgument("Draft length cannot be 0".into()));
        }
        let mut tokens = Vec::with_capacity(draft_length);
        let mut current = seed_token_id;
        for i in 0..draft_length {
            current = DeterministicReferenceOracle::next_decode_step_token(current, 10 + i);
            tokens.push(current);
        }
        Ok(tokens)
    }

    fn verify_tokens(
        &self,
        candidate_tokens: &[u32],
        kv_cache: &DmaBufHandle,
    ) -> Result<usize, EngineError> {
        if candidate_tokens.is_empty() {
            return Ok(0);
        }
        if kv_cache.size() == 0 {
            return Err(EngineError::InvalidArgument("Invalid KV cache buffer size 0".into()));
        }
        let accepted = candidate_tokens.len().min(3);
        Ok(accepted)
    }
}

// ============================================================================
// 6. Memory Watermark Verifier & Zero-Copy Mutation Checker
// ============================================================================

pub struct MemoryWatermarkVerifier;

impl MemoryWatermarkVerifier {
    pub const WATERMARK_MAGIC: u32 = 0x5A_C0_FFEE;

    pub fn fill_watermark(buffer: &mut [u8], seed: u32) {
        for (i, chunk) in buffer.chunks_exact_mut(4).enumerate() {
            let val = seed ^ (i as u32).wrapping_mul(0x1F1F1F1F) ^ Self::WATERMARK_MAGIC;
            chunk.copy_from_slice(&val.to_ne_bytes());
        }
    }

    pub fn verify_watermark(buffer: &[u8], seed: u32) -> bool {
        for (i, chunk) in buffer.chunks_exact(4).enumerate() {
            let expected = seed ^ (i as u32).wrapping_mul(0x1F1F1F1F) ^ Self::WATERMARK_MAGIC;
            if chunk != expected.to_ne_bytes() {
                return false;
            }
        }
        true
    }

    pub fn assert_64b_alignment<T>(ptr: *const T) {
        let addr = ptr as usize;
        assert_eq!(
            addr % 64,
            0,
            "Address 0x{:x} violates strict 64-byte cache line alignment requirement (offset {})",
            addr,
            addr % 64
        );
    }
}

// ============================================================================
// 7. Authoritative Deterministic Reference Oracle (Llama-3-8B Baseline)
// ============================================================================

pub struct DeterministicReferenceOracle;

impl DeterministicReferenceOracle {
    pub const EOS_TOKEN_ID: u32 = 128001; // Llama-3 <|end_of_text|>

    pub const REFERENCE_PROMPT: &'static [u32] = &[128000, 791, 7421, 315, 9607, 374];

    pub const EXPECTED_DECODE_STEPS: &'static [u32] =
        &[9607, 374, 9552, 315, 420, 8496, 11, 7176, 13, 128001];

    pub fn next_token(prompt_tokens: &[u32]) -> u32 {
        if prompt_tokens == Self::REFERENCE_PROMPT {
            Self::EXPECTED_DECODE_STEPS[0]
        } else {
            let hash = prompt_tokens.iter().fold(17u32, |acc, &t| acc.wrapping_mul(31).wrapping_add(t));
            (hash % 100_000) + 1
        }
    }

    pub fn next_decode_step_token(_current_token: u32, sequence_index: usize) -> u32 {
        let step = sequence_index.saturating_sub(Self::REFERENCE_PROMPT.len()) + 1;
        if step < Self::EXPECTED_DECODE_STEPS.len() {
            Self::EXPECTED_DECODE_STEPS[step]
        } else {
            Self::EOS_TOKEN_ID
        }
    }
}
