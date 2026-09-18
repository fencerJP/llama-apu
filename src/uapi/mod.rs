// SPDX-License-Identifier: Apache-2.0
//! Low-level Linux Kernel UAPI bindings for DMA-BUF, DRM Syncobj, AMDXDNA, and AMDGPU.

// -----------------------------------------------------------------------------
// Linux DMA-BUF UAPI (<linux/dma-buf.h>)
// -----------------------------------------------------------------------------

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

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct dma_buf_export_sync_file {
    pub flags: u32,
    pub fd: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct dma_buf_import_sync_file {
    pub flags: u32,
    pub fd: i32,
}

pub const DMA_BUF_BASE: u8 = b'b';

// _IOW('b', 0, struct dma_buf_sync)
nix::ioctl_write_ptr!(dma_buf_ioctl_sync, DMA_BUF_BASE, 0, dma_buf_sync);
nix::ioctl_readwrite!(dma_buf_export_sync_file_ioctl, DMA_BUF_BASE, 2, dma_buf_export_sync_file);
nix::ioctl_write_ptr!(dma_buf_import_sync_file_ioctl, DMA_BUF_BASE, 3, dma_buf_import_sync_file);

// -----------------------------------------------------------------------------
// Linux DRM Syncobj UAPI (<drm/drm.h>)
// -----------------------------------------------------------------------------

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

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_destroy {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_handle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
    pub pad: u32,
    pub point: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_timeline_wait {
    pub handles: u64,       // Pointer to array of __u32 handles
    pub points: u64,        // Pointer to array of __u64 timeline points
    pub timeout_nsec: i64,  // Absolute monotonic timeout in nanoseconds
    pub count_handles: u32, // Number of handles in array
    pub flags: u32,         // DRM_SYNCOBJ_WAIT_FLAGS_*
    pub first_signaled: u32,// Out: index of first signaled handle
    pub pad: u32,
    pub deadline_nsec: u64, // CLOCK_MONOTONIC deadline hint
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_transfer {
    pub src_handle: u32,
    pub dst_handle: u32,
    pub src_point: u64,
    pub dst_point: u64,
    pub flags: u32,
    pub pad: u32,
}

pub const DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED: u32 = 1 << 0;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct drm_syncobj_timeline_array {
    pub handles: u64,       // Pointer to __u32 array
    pub points: u64,        // Pointer to __u64 array
    pub count_handles: u32,
    pub flags: u32,
}

// DRM Core IOCTL base
pub const DRM_IOCTL_BASE: u8 = b'd';
nix::ioctl_readwrite!(drm_syncobj_create_ioctl, DRM_IOCTL_BASE, 0xBF, drm_syncobj_create);
nix::ioctl_readwrite!(drm_syncobj_destroy_ioctl, DRM_IOCTL_BASE, 0xC0, drm_syncobj_destroy);
nix::ioctl_readwrite!(drm_syncobj_handle_to_fd_ioctl, DRM_IOCTL_BASE, 0xC1, drm_syncobj_handle);
nix::ioctl_readwrite!(drm_syncobj_fd_to_handle_ioctl, DRM_IOCTL_BASE, 0xC2, drm_syncobj_handle);
nix::ioctl_readwrite!(drm_syncobj_timeline_wait_ioctl, DRM_IOCTL_BASE, 0xCA, drm_syncobj_timeline_wait);
nix::ioctl_readwrite!(drm_syncobj_query_ioctl, DRM_IOCTL_BASE, 0xCB, drm_syncobj_timeline_array);
nix::ioctl_readwrite!(drm_syncobj_transfer_ioctl, DRM_IOCTL_BASE, 0xCC, drm_syncobj_transfer);
nix::ioctl_readwrite!(drm_syncobj_timeline_signal_ioctl, DRM_IOCTL_BASE, 0xCD, drm_syncobj_timeline_array);

// -----------------------------------------------------------------------------
// AMD XDNA Driver UAPI (<drm/amdxdna_accel.h>)
// -----------------------------------------------------------------------------

pub const DRM_COMMAND_BASE: u8 = 0x40;

pub const DRM_AMDXDNA_CREATE_HWCTX: u8 = 0x00;
pub const DRM_AMDXDNA_DESTROY_HWCTX: u8 = 0x01;
pub const DRM_AMDXDNA_CONFIG_HWCTX: u8 = 0x02;
pub const DRM_AMDXDNA_CREATE_BO: u8 = 0x03;
pub const DRM_AMDXDNA_GET_BO_INFO: u8 = 0x04;
pub const DRM_AMDXDNA_SYNC_BO: u8 = 0x05;
pub const DRM_AMDXDNA_EXEC_CMD: u8 = 0x06;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct amdxdna_drm_create_hwctx {
    pub ext: u64,
    pub ext_flags: u64,
    pub qos_p: u64,
    pub umq_bo: u32,
    pub log_buf_bo: u32,
    pub max_opc: u32,
    pub num_tiles: u32,
    pub mem_size: u32,
    pub umq_doorbell: u32,
    pub handle: u32,
    pub syncobj_handle: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct amdxdna_drm_destroy_hwctx {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct amdxdna_drm_create_bo {
    pub flags: u64,
    pub vaddr: u64,
    pub size: u64,
    pub r#type: u32,
    pub handle: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct amdxdna_drm_va_entry {
    pub vaddr: u64,
    pub len: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct amdxdna_drm_va_tbl {
    pub dmabuf_fd: i32,
    pub num_entries: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct amdxdna_drm_exec_cmd {
    pub ext: u64,
    pub ext_flags: u64,
    pub hwctx: u32,
    pub r#type: u32,
    pub cmd_handles: u64,
    pub args: u64,
    pub cmd_count: u32,
    pub arg_count: u32,
    pub seq: u64,
}

nix::ioctl_readwrite!(amdxdna_create_hwctx_ioctl, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_HWCTX, amdxdna_drm_create_hwctx);
nix::ioctl_readwrite!(amdxdna_destroy_hwctx_ioctl, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_DESTROY_HWCTX, amdxdna_drm_destroy_hwctx);
nix::ioctl_readwrite!(amdxdna_create_bo_ioctl, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_BO, amdxdna_drm_create_bo);
nix::ioctl_readwrite!(amdxdna_exec_cmd_ioctl, DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_EXEC_CMD, amdxdna_drm_exec_cmd);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct amdxdna_drm_get_bo_info {
    pub ext: u64,
    pub ext_flags: u64,
    pub handle: u32,
    pub pad: u32,
    pub map_offset: u64,
    pub vaddr: u64,
    pub xdna_addr: u64,
}
nix::ioctl_readwrite!(
    amdxdna_get_bo_info_ioctl,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_AMDXDNA_GET_BO_INFO,
    amdxdna_drm_get_bo_info
);

// -----------------------------------------------------------------------------
// AMDGPU GEM Driver UAPI (<drm/amdgpu_drm.h>)
// -----------------------------------------------------------------------------

pub const AMDGPU_GEM_DOMAIN_CPU: u64 = 0x1;
pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_DOMAIN_VRAM: u64 = 0x4;
pub const AMDGPU_GEM_DOMAIN_GDS: u64 = 0x8;
pub const AMDGPU_GEM_DOMAIN_GWS: u64 = 0x10;
pub const AMDGPU_GEM_DOMAIN_OA: u64 = 0x20;
pub const AMDGPU_GEM_DOMAIN_MASK: u64 = 0x3f;

pub const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
pub const AMDGPU_GEM_CREATE_NO_CPU_ACCESS: u64 = 1 << 1;
pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub const AMDGPU_GEM_CREATE_VRAM_CLEARED: u64 = 1 << 3;
pub const AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS: u64 = 1 << 5;
pub const AMDGPU_GEM_CREATE_VM_ALWAYS_VALID: u64 = 1 << 6;
pub const AMDGPU_GEM_CREATE_EXPLICIT_SYNC: u64 = 1 << 7;
pub const AMDGPU_GEM_CREATE_CP_MQD_GFX9: u64 = 1 << 8;
pub const AMDGPU_GEM_CREATE_VRAM_WIPE_ON_RELEASE: u64 = 1 << 9;
pub const AMDGPU_GEM_CREATE_ENCRYPTED: u64 = 1 << 10;
pub const AMDGPU_GEM_CREATE_PREEMPTIBLE: u64 = 1 << 11;
pub const AMDGPU_GEM_CREATE_DISCARDABLE: u64 = 1 << 12;
pub const AMDGPU_GEM_CREATE_COHERENT: u64 = 1 << 13;
pub const AMDGPU_GEM_CREATE_UNCACHED: u64 = 1 << 14;
pub const AMDGPU_GEM_CREATE_EXT_COHERENT: u64 = 1 << 15;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct drm_amdgpu_gem_create_in {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
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

impl Default for drm_amdgpu_gem_create {
    fn default() -> Self {
        Self {
            r#in: drm_amdgpu_gem_create_in::default(),
        }
    }
}

pub const DRM_AMDGPU_GEM_CREATE: u8 = 0x00;
nix::ioctl_readwrite!(
    amdgpu_gem_create_ioctl,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE,
    drm_amdgpu_gem_create
);

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct drm_gem_close {
    pub handle: u32,
    pub pad: u32,
}
nix::ioctl_write_ptr!(drm_gem_close_ioctl, DRM_IOCTL_BASE, 0x09, drm_gem_close);

// -----------------------------------------------------------------------------
// DRM PRIME Interop (<drm/drm.h>)
// -----------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct drm_prime_handle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
}

pub const DRM_CLOEXEC: u32 = 0x80000;
pub const DRM_RDWR: u32 = 0x00002;

nix::ioctl_readwrite!(drm_prime_handle_to_fd_ioctl, DRM_IOCTL_BASE, 0x2D, drm_prime_handle);
nix::ioctl_readwrite!(drm_prime_fd_to_handle_ioctl, DRM_IOCTL_BASE, 0x2E, drm_prime_handle);

// -----------------------------------------------------------------------------
// Compile-time Layout & Size Assertions (<drm/drm.h>, <linux/dma-buf.h>, etc.)
// -----------------------------------------------------------------------------

const _: () = {
    assert!(core::mem::size_of::<dma_buf_sync>() == 8);
    assert!(core::mem::size_of::<dma_buf_export_sync_file>() == 8);
    assert!(core::mem::size_of::<dma_buf_import_sync_file>() == 8);
    assert!(core::mem::size_of::<drm_syncobj_create>() == 8);
    assert!(core::mem::size_of::<drm_syncobj_destroy>() == 8);
    assert!(core::mem::size_of::<drm_syncobj_handle>() == 24);
    assert!(core::mem::offset_of!(drm_syncobj_handle, point) == 16);
    assert!(core::mem::size_of::<drm_syncobj_timeline_wait>() == 48);
    assert!(core::mem::size_of::<drm_syncobj_timeline_array>() == 24);
    assert!(core::mem::size_of::<drm_syncobj_transfer>() == 32);
    assert!(core::mem::size_of::<drm_prime_handle>() == 12);
    assert!(core::mem::size_of::<drm_gem_close>() == 8);
    assert!(core::mem::size_of::<drm_amdgpu_gem_create_in>() == 32);
    assert!(core::mem::size_of::<drm_amdgpu_gem_create_out>() == 8);
    assert!(core::mem::size_of::<drm_amdgpu_gem_create>() == 32);
    assert!(core::mem::size_of::<amdxdna_drm_create_hwctx>() == 56);
    assert!(core::mem::size_of::<amdxdna_drm_destroy_hwctx>() == 8);
    assert!(core::mem::size_of::<amdxdna_drm_create_bo>() == 32);
    assert!(core::mem::size_of::<amdxdna_drm_va_entry>() == 16);
    assert!(core::mem::size_of::<amdxdna_drm_va_tbl>() == 8);
    assert!(core::mem::size_of::<amdxdna_drm_get_bo_info>() == 48);
    assert!(core::mem::size_of::<amdxdna_drm_exec_cmd>() == 56);
};

// -----------------------------------------------------------------------------
// IOCTL Encoding Helpers & Constants (_IOC macro emulation)
// -----------------------------------------------------------------------------

pub const _IOC_NONE: u64 = 0;
pub const _IOC_WRITE: u64 = 1;
pub const _IOC_READ: u64 = 2;
pub const _IOC_READWRITE: u64 = 3;

#[inline]
pub const fn _ioc(dir: u64, r#type: u8, nr: u8, size: usize) -> u64 {
    (dir << 30) | ((size as u64) << 16) | ((r#type as u64) << 8) | (nr as u64)
}

#[inline]
pub const fn _iow<T>(r#type: u8, nr: u8) -> u64 {
    _ioc(_IOC_WRITE, r#type, nr, core::mem::size_of::<T>())
}

#[inline]
pub const fn _iowr<T>(r#type: u8, nr: u8) -> u64 {
    _ioc(_IOC_READWRITE, r#type, nr, core::mem::size_of::<T>())
}

pub const DRM_IOCTL_AMDGPU_GEM_CREATE_NUM: u64 = _iowr::<drm_amdgpu_gem_create>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE);
pub const DRM_IOCTL_GEM_CLOSE_NUM: u64 = _iow::<drm_gem_close>(DRM_IOCTL_BASE, 0x09);
pub const DRM_IOCTL_PRIME_HANDLE_TO_FD_NUM: u64 = _iowr::<drm_prime_handle>(DRM_IOCTL_BASE, 0x2D);
pub const DRM_IOCTL_PRIME_FD_TO_HANDLE_NUM: u64 = _iowr::<drm_prime_handle>(DRM_IOCTL_BASE, 0x2E);
pub const DRM_IOCTL_SYNCOBJ_CREATE_NUM: u64 = _iowr::<drm_syncobj_create>(DRM_IOCTL_BASE, 0xBF);
pub const DRM_IOCTL_SYNCOBJ_DESTROY_NUM: u64 = _iowr::<drm_syncobj_destroy>(DRM_IOCTL_BASE, 0xC0);
pub const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM: u64 = _iowr::<drm_syncobj_handle>(DRM_IOCTL_BASE, 0xC1);
pub const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_NUM: u64 = _iowr::<drm_syncobj_handle>(DRM_IOCTL_BASE, 0xC2);
pub const DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM: u64 = _iowr::<drm_syncobj_timeline_wait>(DRM_IOCTL_BASE, 0xCA);
pub const DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM: u64 = _iowr::<drm_syncobj_timeline_array>(DRM_IOCTL_BASE, 0xCD);
pub const DRM_IOCTL_SYNCOBJ_TRANSFER_NUM: u64 = _iowr::<drm_syncobj_transfer>(DRM_IOCTL_BASE, 0xCC);
pub const DRM_IOCTL_AMDXDNA_CREATE_BO_NUM: u64 = _iowr::<amdxdna_drm_create_bo>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_BO);
pub const DRM_IOCTL_AMDXDNA_GET_BO_INFO_NUM: u64 = _iowr::<amdxdna_drm_get_bo_info>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDXDNA_GET_BO_INFO);
pub const DMA_BUF_IOCTL_SYNC_NUM: u64 = _iow::<dma_buf_sync>(DMA_BUF_BASE, 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uapi_struct_sizes() {
        assert_eq!(core::mem::size_of::<drm_syncobj_handle>(), 24);
        assert_eq!(core::mem::size_of::<drm_syncobj_timeline_wait>(), 48);
        assert_eq!(core::mem::size_of::<drm_syncobj_timeline_array>(), 24);
        assert_eq!(core::mem::size_of::<drm_syncobj_transfer>(), 32);
        assert_eq!(core::mem::size_of::<drm_amdgpu_gem_create>(), 32);
        assert_eq!(core::mem::size_of::<drm_gem_close>(), 8);
        assert_eq!(core::mem::size_of::<drm_prime_handle>(), 12);
        assert_eq!(core::mem::size_of::<amdxdna_drm_get_bo_info>(), 48);
        assert_eq!(core::mem::size_of::<amdxdna_drm_create_bo>(), 32);
    }

    #[test]
    fn test_uapi_field_offsets() {
        assert_eq!(core::mem::offset_of!(drm_syncobj_handle, point), 16);
        assert_eq!(core::mem::offset_of!(drm_syncobj_timeline_wait, deadline_nsec), 40);
        assert_eq!(core::mem::offset_of!(amdxdna_drm_get_bo_info, xdna_addr), 40);
    }

    #[test]
    fn test_uapi_ioctl_numbers() {
        assert_eq!(DRM_IOCTL_AMDGPU_GEM_CREATE_NUM, 0xc0206440);
        assert_eq!(DRM_IOCTL_GEM_CLOSE_NUM, 0x40086409);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD_NUM, 0xc00c642d);
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE_NUM, 0xc00c642e);
        assert_eq!(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM, 0xc01864c1);
        assert_eq!(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE_NUM, 0xc01864c2);
        assert_eq!(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM, 0xc03064ca);
        assert_eq!(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM, 0xc01864cd);
        assert_eq!(DRM_IOCTL_AMDXDNA_CREATE_BO_NUM, 0xc0206443);
        assert_eq!(DRM_IOCTL_AMDXDNA_GET_BO_INFO_NUM, 0xc0306444);
        assert_eq!(DMA_BUF_IOCTL_SYNC_NUM, 0x40086200);
    }
}
