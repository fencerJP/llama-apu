// SPDX-License-Identifier: Apache-2.0
//! Device backend abstractions for AMDGPU and AMDXDNA accelerators.
//!
//! Provides trait-based unified access to physical silicon (`/dev/dri/renderD128`,
//! `/dev/accel/accel0`) and in-memory mock devices for deterministic, rootless testing.

use std::os::fd::RawFd;
use std::sync::Arc;
use thiserror::Error;

pub mod mock;
pub mod physical;

pub use mock::MockDeviceBackend;
pub use physical::PhysicalDeviceBackend;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Nix error: {0}")]
    Nix(#[from] nix::Error),
    #[error("Device error: {0}")]
    DeviceError(String),
}

/// Device classification for accelerators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// RDNA iGPU render node (e.g. `/dev/dri/renderD128`)
    Gpu,
    /// XDNA NPU compute node (e.g. `/dev/accel/accel0`)
    Npu,
}

/// Unified abstraction for accelerator device nodes.
pub trait DeviceBackend: Send + Sync + std::fmt::Debug {
    /// Issue an ioctl call against the underlying device node or virtual shim.
    fn ioctl(&self, request: u64, arg: *mut libc::c_void) -> Result<i32, nix::Error>;

    /// Borrow the underlying raw file descriptor.
    fn as_raw_fd(&self) -> RawFd;

    /// Return whether this backend is an emulated mock driver.
    fn is_mock(&self) -> bool {
        false
    }

    /// Return the device type (GPU or NPU).
    fn device_type(&self) -> DeviceType;
}

/// Automatic device discovery and instantiation factory.
///
/// Attempts to open the physical device node at `path`. If physical silicon is
/// absent or inaccessible, gracefully falls back to a high-fidelity `MockDeviceBackend`.
pub fn open_or_mock(path: &str) -> Result<Arc<dyn DeviceBackend>, BackendError> {
    let dev_type = if path.contains("accel") || path.contains("npu") {
        DeviceType::Npu
    } else {
        DeviceType::Gpu
    };

    match PhysicalDeviceBackend::open(path, dev_type) {
        Ok(backend) => {
            tracing::info!(path = path, "Connected to physical accelerator silicon");
            Ok(Arc::new(backend))
        }
        Err(err) => {
            tracing::warn!(
                path = path,
                error = %err,
                "Physical silicon unavailable; falling back to MockDeviceBackend"
            );
            Ok(Arc::new(MockDeviceBackend::new(dev_type)))
        }
    }
}

impl dyn DeviceBackend {
    /// Associated factory on `dyn DeviceBackend` mirroring `open_or_mock`.
    pub fn open_or_mock(path: &str) -> Result<Arc<dyn DeviceBackend>, BackendError> {
        open_or_mock(path)
    }
}
