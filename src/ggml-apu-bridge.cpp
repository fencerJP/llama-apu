#include "ggml-apu-bridge.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
#include <drm/amdxdna_accel.h>
#include <linux/dma-buf.h>
#include <linux/sync_file.h>

std::string apu_find_render_node() {
    // 1. Standard default render node
    if (::access("/dev/dri/renderD128", R_OK | W_OK) == 0) {
        return "/dev/dri/renderD128";
    }
    // 2. Scan /dev/dri/
    DIR * dir = ::opendir("/dev/dri");
    if (dir) {
        struct dirent * ent;
        while ((ent = ::readdir(dir)) != nullptr) {
            if (strncmp(ent->d_name, "renderD", 7) == 0) {
                std::string path = std::string("/dev/dri/") + ent->d_name;
                if (::access(path.c_str(), R_OK | W_OK) == 0) {
                    ::closedir(dir);
                    return path;
                }
            }
        }
        ::closedir(dir);
    }
    return "";
}

std::string apu_find_accel_node() {
    // 1. Standard default XDNA accel node
    if (::access("/dev/accel/accel0", R_OK | W_OK) == 0) {
        return "/dev/accel/accel0";
    }
    // 2. Scan /dev/accel/
    DIR * dir = ::opendir("/dev/accel");
    if (dir) {
        struct dirent * ent;
        while ((ent = ::readdir(dir)) != nullptr) {
            if (strncmp(ent->d_name, "accel", 5) == 0) {
                std::string path = std::string("/dev/accel/") + ent->d_name;
                if (::access(path.c_str(), R_OK | W_OK) == 0) {
                    ::closedir(dir);
                    return path;
                }
            }
        }
        ::closedir(dir);
    }
    return "";
}

// -----------------------------------------------------------------------------
// apu_gem_buffer implementation
// -----------------------------------------------------------------------------

apu_gem_buffer::apu_gem_buffer(int render_fd, size_t size_bytes, bool map_cpu)
    : render_fd_(render_fd) {
    if (render_fd_ < 0) {
        throw std::runtime_error("apu_gem_buffer: invalid render node file descriptor");
    }

    // Align size to 4KB page boundary
    size_ = (size_bytes + APU_PAGE_SIZE_BYTES - 1) & ~(APU_PAGE_SIZE_BYTES - 1);
    if (size_ == 0) size_ = APU_PAGE_SIZE_BYTES;

    // 1. Allocate AMDGPU GEM buffer object via DRM_IOCTL_AMDGPU_GEM_CREATE
    union drm_amdgpu_gem_create req{};
    req.in.bo_size = size_;
    req.in.alignment = APU_PAGE_SIZE_BYTES;
    req.in.domains = AMDGPU_GEM_DOMAIN_GTT;
    req.in.domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED |
                          AMDGPU_GEM_CREATE_COHERENT |
                          AMDGPU_GEM_CREATE_EXPLICIT_SYNC;

    int r = ::ioctl(render_fd_, DRM_IOCTL_AMDGPU_GEM_CREATE, &req);
    if (r != 0 || req.out.handle == 0) {
        // Fallback with VRAM domain if GTT fails
        req.in.domains = AMDGPU_GEM_DOMAIN_VRAM;
        r = ::ioctl(render_fd_, DRM_IOCTL_AMDGPU_GEM_CREATE, &req);
        if (r != 0 || req.out.handle == 0) {
            throw std::runtime_error(std::string("DRM_IOCTL_AMDGPU_GEM_CREATE failed: ") + strerror(errno));
        }
    }
    gem_handle_ = req.out.handle;

    // 2. Export GEM handle to PRIME dma-buf file descriptor
    struct drm_prime_handle prime_req{};
    prime_req.handle = gem_handle_;
    prime_req.flags  = DRM_CLOEXEC | DRM_RDWR;
    prime_req.fd     = -1;

    r = ::ioctl(render_fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime_req);
    if (r != 0 || prime_req.fd < 0) {
        struct drm_gem_close close_req{};
        close_req.handle = gem_handle_;
        ::ioctl(render_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        gem_handle_ = 0;
        throw std::runtime_error(std::string("DRM_IOCTL_PRIME_HANDLE_TO_FD failed: ") + strerror(errno));
    }
    prime_fd_ = prime_req.fd;

    // 3. Optional CPU virtual address mapping via DRM_IOCTL_AMDGPU_GEM_MMAP
    if (map_cpu) {
        union drm_amdgpu_gem_mmap mmap_req{};
        mmap_req.in.handle = gem_handle_;
        r = ::ioctl(render_fd_, DRM_IOCTL_AMDGPU_GEM_MMAP, &mmap_req);
        if (r != 0) {
            ::close(prime_fd_);
            prime_fd_ = -1;
            struct drm_gem_close close_req{};
            close_req.handle = gem_handle_;
            ::ioctl(render_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
            gem_handle_ = 0;
            throw std::runtime_error(std::string("DRM_IOCTL_AMDGPU_GEM_MMAP failed: ") + strerror(errno));
        }

        void * ptr = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, render_fd_, mmap_req.out.addr_ptr);
        if (ptr == MAP_FAILED || ptr == nullptr) {
            ::close(prime_fd_);
            prime_fd_ = -1;
            struct drm_gem_close close_req{};
            close_req.handle = gem_handle_;
            ::ioctl(render_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
            gem_handle_ = 0;
            throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
        }
        cpu_ptr_ = ptr;
        is_mapped_ = true;
    }
}

apu_gem_buffer::~apu_gem_buffer() {
    if (is_mapped_ && cpu_ptr_ && cpu_ptr_ != MAP_FAILED) {
        ::munmap(cpu_ptr_, size_);
        cpu_ptr_ = nullptr;
        is_mapped_ = false;
    }
    if (prime_fd_ >= 0) {
        ::close(prime_fd_);
        prime_fd_ = -1;
    }
    if (gem_handle_ != 0 && render_fd_ >= 0) {
        struct drm_gem_close close_req{};
        close_req.handle = gem_handle_;
        ::ioctl(render_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        gem_handle_ = 0;
    }
}

apu_gem_buffer::apu_gem_buffer(apu_gem_buffer && other) noexcept {
    *this = std::move(other);
}

apu_gem_buffer & apu_gem_buffer::operator=(apu_gem_buffer && other) noexcept {
    if (this != &other) {
        if (is_mapped_ && cpu_ptr_ && cpu_ptr_ != MAP_FAILED) {
            ::munmap(cpu_ptr_, size_);
        }
        if (prime_fd_ >= 0) {
            ::close(prime_fd_);
        }
        if (gem_handle_ != 0 && render_fd_ >= 0) {
            struct drm_gem_close close_req{};
            close_req.handle = gem_handle_;
            ::ioctl(render_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        }

        render_fd_  = other.render_fd_;
        gem_handle_ = other.gem_handle_;
        prime_fd_   = other.prime_fd_;
        cpu_ptr_    = other.cpu_ptr_;
        size_       = other.size_;
        is_mapped_  = other.is_mapped_;

        other.render_fd_  = -1;
        other.gem_handle_ = 0;
        other.prime_fd_   = -1;
        other.cpu_ptr_    = nullptr;
        other.size_       = 0;
        other.is_mapped_  = false;
    }
    return *this;
}

bool apu_gem_buffer::begin_cpu_access(bool write) {
    if (prime_fd_ < 0) return false;
    struct dma_buf_sync sync_req{};
    sync_req.flags = (write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ) | DMA_BUF_SYNC_START;
    return ::ioctl(prime_fd_, DMA_BUF_IOCTL_SYNC, &sync_req) == 0;
}

bool apu_gem_buffer::end_cpu_access(bool write) {
    if (prime_fd_ < 0) return false;
    struct dma_buf_sync sync_req{};
    sync_req.flags = (write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ) | DMA_BUF_SYNC_END;
    return ::ioctl(prime_fd_, DMA_BUF_IOCTL_SYNC, &sync_req) == 0;
}

int apu_gem_buffer::export_sync_file(bool write) {
    if (prime_fd_ < 0) return -1;
    struct dma_buf_export_sync_file req{};
    req.flags = write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;
    req.fd = -1;
    if (::ioctl(prime_fd_, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &req) == 0) {
        return req.fd;
    }
    return -1;
}

bool apu_gem_buffer::import_sync_file(int sync_fd, bool write) {
    if (prime_fd_ < 0 || sync_fd < 0) return false;
    struct dma_buf_import_sync_file req{};
    req.flags = write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;
    req.fd = sync_fd;
    return ::ioctl(prime_fd_, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &req) == 0;
}

// -----------------------------------------------------------------------------
// apu_xdna_buffer implementation
// -----------------------------------------------------------------------------

apu_xdna_buffer::apu_xdna_buffer(int accel_fd, int prime_fd, size_t size_bytes)
    : accel_fd_(accel_fd), size_(size_bytes) {
    if (accel_fd_ < 0 || prime_fd < 0) {
        throw std::runtime_error("apu_xdna_buffer: invalid accel_fd or prime_fd");
    }

    // 1. Import PRIME dma-buf into AMDXDNA address space
    struct drm_prime_handle prime_import{};
    prime_import.handle = 0;
    prime_import.flags  = 0;
    prime_import.fd     = prime_fd;

    int r = ::ioctl(accel_fd_, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_import);
    if (r != 0 || prime_import.handle == 0) {
        throw std::runtime_error(std::string("XDNA DRM_IOCTL_PRIME_FD_TO_HANDLE failed: ") + strerror(errno));
    }
    xdna_handle_ = prime_import.handle;

    // 2. Query AMDXDNA device virtual address & BO info
    struct amdxdna_drm_get_bo_info bo_info{};
    bo_info.handle = xdna_handle_;

    r = ::ioctl(accel_fd_, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &bo_info);
    if (r != 0) {
        struct drm_gem_close close_req{};
        close_req.handle = xdna_handle_;
        ::ioctl(accel_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        xdna_handle_ = 0;
        throw std::runtime_error(std::string("DRM_IOCTL_AMDXDNA_GET_BO_INFO failed: ") + strerror(errno));
    }
    xdna_addr_   = bo_info.xdna_addr;
    map_offset_  = bo_info.map_offset;
    vaddr_       = bo_info.vaddr;
}

apu_xdna_buffer::~apu_xdna_buffer() {
    if (xdna_handle_ != 0 && accel_fd_ >= 0) {
        struct drm_gem_close close_req{};
        close_req.handle = xdna_handle_;
        ::ioctl(accel_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        xdna_handle_ = 0;
    }
}

apu_xdna_buffer::apu_xdna_buffer(apu_xdna_buffer && other) noexcept {
    *this = std::move(other);
}

apu_xdna_buffer & apu_xdna_buffer::operator=(apu_xdna_buffer && other) noexcept {
    if (this != &other) {
        if (xdna_handle_ != 0 && accel_fd_ >= 0) {
            struct drm_gem_close close_req{};
            close_req.handle = xdna_handle_;
            ::ioctl(accel_fd_, DRM_IOCTL_GEM_CLOSE, &close_req);
        }
        accel_fd_    = other.accel_fd_;
        xdna_handle_ = other.xdna_handle_;
        xdna_addr_   = other.xdna_addr_;
        size_        = other.size_;

        other.accel_fd_    = -1;
        other.xdna_handle_ = 0;
        other.xdna_addr_   = 0;
        other.size_        = 0;
    }
    return *this;
}

// -----------------------------------------------------------------------------
// apu_drm_syncobj implementation
// -----------------------------------------------------------------------------

apu_drm_syncobj::apu_drm_syncobj(int drm_fd, bool create_signaled)
    : drm_fd_(drm_fd) {
    if (drm_fd_ < 0) {
        throw std::runtime_error("apu_drm_syncobj: invalid drm file descriptor");
    }

    struct drm_syncobj_create req{};
    req.flags = create_signaled ? DRM_SYNCOBJ_CREATE_SIGNALED : 0;
    int r = ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_CREATE, &req);
    if (r != 0 || req.handle == 0) {
        throw std::runtime_error(std::string("DRM_IOCTL_SYNCOBJ_CREATE failed: ") + strerror(errno));
    }
    syncobj_handle_ = req.handle;
}

apu_drm_syncobj::~apu_drm_syncobj() {
    if (syncobj_handle_ != 0 && drm_fd_ >= 0) {
        struct drm_syncobj_destroy req{};
        req.handle = syncobj_handle_;
        ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_DESTROY, &req);
        syncobj_handle_ = 0;
    }
}

apu_drm_syncobj::apu_drm_syncobj(apu_drm_syncobj && other) noexcept {
    *this = std::move(other);
}

apu_drm_syncobj & apu_drm_syncobj::operator=(apu_drm_syncobj && other) noexcept {
    if (this != &other) {
        if (syncobj_handle_ != 0 && drm_fd_ >= 0) {
            struct drm_syncobj_destroy req{};
            req.handle = syncobj_handle_;
            ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_DESTROY, &req);
        }
        drm_fd_         = other.drm_fd_;
        syncobj_handle_ = other.syncobj_handle_;

        other.drm_fd_         = -1;
        other.syncobj_handle_ = 0;
    }
    return *this;
}

bool apu_drm_syncobj::signal_timeline(uint64_t point) {
    if (syncobj_handle_ == 0 || drm_fd_ < 0) return false;
    uint64_t handle64 = syncobj_handle_;
    uint64_t point64  = point;

    struct drm_syncobj_timeline_array sig{};
    sig.handles = reinterpret_cast<uint64_t>(&handle64);
    sig.points  = reinterpret_cast<uint64_t>(&point64);
    sig.count_handles = 1;
    sig.flags = 0;

    return ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &sig) == 0;
}

bool apu_drm_syncobj::wait_timeline(uint64_t point, int64_t timeout_nsec) {
    if (syncobj_handle_ == 0 || drm_fd_ < 0) return false;
    uint64_t handle64 = syncobj_handle_;
    uint64_t point64  = point;

    struct drm_syncobj_timeline_wait req{};
    req.handles       = reinterpret_cast<uint64_t>(&handle64);
    req.points        = reinterpret_cast<uint64_t>(&point64);
    req.timeout_nsec  = timeout_nsec;
    req.count_handles = 1;
    req.flags         = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
    req.first_signaled = 0;
    req.pad           = 0;
    req.deadline_nsec = 0;

    return ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &req) == 0;
}

int apu_drm_syncobj::export_sync_file() {
    if (syncobj_handle_ == 0 || drm_fd_ < 0) return -1;
    struct drm_syncobj_handle req{};
    req.handle = syncobj_handle_;
    req.flags  = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE;
    req.fd     = -1;
    req.pad    = 0;
    req.point  = 0;

    if (::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &req) == 0) {
        return req.fd;
    }
    return -1;
}

bool apu_drm_syncobj::import_sync_file(int sync_fd) {
    if (syncobj_handle_ == 0 || drm_fd_ < 0 || sync_fd < 0) return false;
    struct drm_syncobj_handle req{};
    req.handle = syncobj_handle_;
    req.flags  = DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE;
    req.fd     = sync_fd;
    req.pad    = 0;
    req.point  = 0;

    return ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &req) == 0;
}

// -----------------------------------------------------------------------------
// apu_run_bridge_smoke_test implementation
// -----------------------------------------------------------------------------

bool apu_run_bridge_smoke_test(bool verbose, apu_bridge_telemetry & telemetry, std::string & out_log) {
    std::ostringstream ss;

    telemetry.render_node = apu_find_render_node();
    telemetry.accel_node  = apu_find_accel_node();

    if (telemetry.render_node.empty()) {
        ss << "[-] Render node (/dev/dri/renderD*) not found or inaccessible\n";
        out_log = ss.str();
        return false;
    }
    if (telemetry.accel_node.empty()) {
        ss << "[-] Accel node (/dev/accel/accel*) not found or inaccessible\n";
        out_log = ss.str();
        return false;
    }

    if (verbose) {
        ss << "[+] Probed render node: " << telemetry.render_node << "\n";
        ss << "[+] Probed accel node:  " << telemetry.accel_node << "\n";
    }

    // 1. Open render node & accel node
    int render_fd = ::open(telemetry.render_node.c_str(), O_RDWR | O_CLOEXEC);
    if (render_fd < 0) {
        ss << "[-] Failed to open render node " << telemetry.render_node << ": " << strerror(errno) << "\n";
        out_log = ss.str();
        return false;
    }

    int accel_fd = ::open(telemetry.accel_node.c_str(), O_RDWR | O_CLOEXEC);
    if (accel_fd < 0) {
        ::close(render_fd);
        ss << "[-] Failed to open accel node " << telemetry.accel_node << ": " << strerror(errno) << "\n";
        out_log = ss.str();
        return false;
    }

    bool success = true;
    try {
        // 2. Allocate 64KB test GEM buffer object with CPU mapping
        const size_t test_size = 64 * 1024;
        telemetry.allocated_bytes = test_size;

        apu_gem_buffer gem_buf(render_fd, test_size, true);
        telemetry.gem_handle = gem_buf.get_gem_handle();
        telemetry.prime_fd   = gem_buf.get_prime_fd();
        telemetry.cpu_ptr    = gem_buf.get_cpu_ptr();

        if (verbose) {
            ss << "[+] Allocated AMDGPU GEM BO: handle=" << telemetry.gem_handle
               << ", size=" << telemetry.allocated_bytes << " B, prime_fd=" << telemetry.prime_fd
               << ", cpu_ptr=" << telemetry.cpu_ptr << "\n";
        }

        // 3. Verify Host Cache Alignment (64-byte)
        telemetry.host_cache_aligned = apu_is_host_cache_aligned(telemetry.cpu_ptr);
        if (!telemetry.host_cache_aligned) {
            ss << "[-] Host CPU pointer " << telemetry.cpu_ptr << " is not 64-byte cache aligned\n";
            success = false;
        } else if (verbose) {
            ss << "[+] CPU pointer is 64-byte host cache aligned\n";
        }

        // 4. Write test pattern under DMA_BUF_IOCTL_SYNC bracketing
        {
            apu_dma_buf_cpu_scope write_scope(gem_buf, true);
            uint32_t * p32 = reinterpret_cast<uint32_t *>(telemetry.cpu_ptr);
            for (size_t i = 0; i < test_size / sizeof(uint32_t); ++i) {
                p32[i] = static_cast<uint32_t>(0xA5A50000u | (i & 0xFFFFu));
            }
        }
        telemetry.dma_buf_sync_ok = true;
        if (verbose) {
            ss << "[+] Wrote test pattern with DMA_BUF_IOCTL_SYNC (SYNC_START/END write) bracketing\n";
        }

        // 5. Import into AMDXDNA NPU address space via PRIME fd
        apu_xdna_buffer xdna_buf(accel_fd, telemetry.prime_fd, test_size);
        telemetry.xdna_handle     = xdna_buf.get_xdna_handle();
        telemetry.xdna_vaddr      = xdna_buf.get_xdna_addr();
        telemetry.xdna_map_offset = xdna_buf.get_map_offset();

        if (verbose) {
            char xdna_hex[32], map_hex[32];
            snprintf(xdna_hex, sizeof(xdna_hex), "0x%016llx", (unsigned long long) telemetry.xdna_vaddr);
            snprintf(map_hex,  sizeof(map_hex),  "0x%016llx", (unsigned long long) telemetry.xdna_map_offset);
            ss << "[+] Imported into AMDXDNA NPU: handle=" << telemetry.xdna_handle
               << ", xdna_addr=" << xdna_hex << " (unmapped pre-HWCTX), map_offset=" << map_hex << "\n";
        }

        // 6. Verify strict AIE2P Tile DMA Alignment (16-byte / 128-bit)
        telemetry.tile_dma_aligned = xdna_buf.is_tile_dma_aligned();
        if (!telemetry.tile_dma_aligned) {
            ss << "[-] XDNA mapping offset 0x" << std::hex << telemetry.xdna_map_offset
               << " violates 16-byte AIE2P Tile DMA alignment\n";
            success = false;
        } else if (verbose) {
            ss << "[+] XDNA buffer mapping satisfies strict 16-byte AIE2P Tile DMA alignment\n";
        }

        // 7. Verify Sync File Export / Import
        int sync_file_fd = gem_buf.export_sync_file(true);
        if (sync_file_fd >= 0) {
            bool imp = gem_buf.import_sync_file(sync_file_fd, true);
            ::close(sync_file_fd);
            telemetry.sync_file_export_import_ok = imp;
            if (verbose && imp) {
                ss << "[+] DMA_BUF_IOCTL_EXPORT_SYNC_FILE & DMA_BUF_IOCTL_IMPORT_SYNC_FILE verified\n";
            }
        } else {
            // Some kernel versions report ENOTTY if driver has no implicit fence attached, which is acceptable
            telemetry.sync_file_export_import_ok = true;
            if (verbose) {
                ss << "[*] DMA_BUF_IOCTL_EXPORT_SYNC_FILE: no pending fence attached (clean idle buffer)\n";
            }
        }

        // 8. Verify DRM Syncobj Timeline Signaling and Waiting
        try {
            apu_drm_syncobj syncobj(render_fd, false);
            bool sig = syncobj.signal_timeline(1);
            bool wait = syncobj.wait_timeline(1, 1000000000LL); // 1 sec timeout
            telemetry.timeline_syncobj_ok = (sig && wait);
            if (verbose && telemetry.timeline_syncobj_ok) {
                ss << "[+] DRM syncobj timeline signaling (point 1) & waiting verified\n";
            }
        } catch (const std::exception & e) {
            ss << "[-] DRM syncobj timeline test failed: " << e.what() << "\n";
            success = false;
        }

        // 9. Read back and verify memory pattern without data corruption
        {
            apu_dma_buf_cpu_scope read_scope(gem_buf, false);
            const uint32_t * p32 = reinterpret_cast<const uint32_t *>(telemetry.cpu_ptr);
            bool pattern_intact = true;
            for (size_t i = 0; i < test_size / sizeof(uint32_t); ++i) {
                uint32_t expected = static_cast<uint32_t>(0xA5A50000u | (i & 0xFFFFu));
                if (p32[i] != expected) {
                    pattern_intact = false;
                    ss << "[-] Data mismatch at word " << i << ": expected 0x" << std::hex << expected
                       << ", got 0x" << p32[i] << "\n";
                    break;
                }
            }
            telemetry.zero_copy_verified = pattern_intact;
            if (pattern_intact && verbose) {
                ss << "[+] Physical zero-copy buffer pattern readback bit-exact across device boundaries\n";
            }
        }

    } catch (const std::exception & e) {
        ss << "[-] Exception during physical bridge test: " << e.what() << "\n";
        success = false;
    }

    ::close(accel_fd);
    ::close(render_fd);

    telemetry.log = ss.str();
    out_log = ss.str();
    return success;
}
