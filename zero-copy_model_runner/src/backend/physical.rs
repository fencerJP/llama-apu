// SPDX-License-Identifier: Apache-2.0
//! Physical accelerator device backend routing directly to Linux character device nodes.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};

use super::{BackendError, DeviceBackend, DeviceType};

/// Backend representing a physical Linux device node (`/dev/dri/renderD*` or `/dev/accel/*`).
#[derive(Debug)]
pub struct PhysicalDeviceBackend {
    fd: OwnedFd,
    path: PathBuf,
    dev_type: DeviceType,
}

impl PhysicalDeviceBackend {
    /// Open a physical device node with `O_RDWR | O_CLOEXEC`.
    pub fn open<P: AsRef<Path>>(path: P, dev_type: DeviceType) -> Result<Self, BackendError> {
        let p = path.as_ref();
        let raw_fd = nix::fcntl::open(
            p,
            nix::fcntl::OFlag::O_RDWR | nix::fcntl::OFlag::O_CLOEXEC,
            nix::sys::stat::Mode::empty(),
        )?;

        Ok(Self {
            fd: unsafe { OwnedFd::from_raw_fd(raw_fd) },
            path: p.to_path_buf(),
            dev_type,
        })
    }

    /// Access the file path used to open this device.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl DeviceBackend for PhysicalDeviceBackend {
    fn ioctl(&self, request: u64, arg: *mut libc::c_void) -> Result<i32, nix::Error> {
        let ret = unsafe { libc::ioctl(self.fd.as_raw_fd(), request, arg) };
        if ret < 0 {
            Err(nix::Error::last())
        } else {
            Ok(ret)
        }
    }

    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    fn is_mock(&self) -> bool {
        false
    }

    fn device_type(&self) -> DeviceType {
        self.dev_type
    }
}
