#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

// Hardware alignment constants per AIE2P (XDNA 2) architecture
constexpr size_t APU_TILE_DMA_ALIGNMENT_BYTES = 16;  // 128-bit Tile DMA transfer beat
constexpr size_t APU_HOST_CACHE_ALIGNMENT_BYTES = 64; // Host cache line alignment
constexpr size_t APU_PAGE_SIZE_BYTES = 4096;          // Standard 4KB page alignment

struct apu_bridge_telemetry {
    size_t      allocated_bytes = 0;
    uint32_t    gem_handle      = 0;
    int         prime_fd        = -1;
    uint32_t    xdna_handle     = 0;
    uint64_t    xdna_vaddr      = 0;
    uint64_t    xdna_map_offset = 0;
    void *      cpu_ptr         = nullptr;
    size_t      base_alignment  = 0;
    bool        tile_dma_aligned = false;
    bool        host_cache_aligned = false;
    bool        dma_buf_sync_ok = false;
    bool        sync_file_export_import_ok = false;
    bool        timeline_syncobj_ok = false;
    bool        zero_copy_verified = false;
    std::string render_node;
    std::string accel_node;
    std::string log;
};

// Probe hardware device nodes
std::string apu_find_render_node();
std::string apu_find_accel_node();

// Check if address/offset satisfies 16-byte AIE2P Tile DMA alignment
inline bool apu_is_tile_dma_aligned(uint64_t addr, size_t offset = 0) {
    return ((addr + offset) % APU_TILE_DMA_ALIGNMENT_BYTES) == 0;
}

// Check if pointer satisfies 64-byte host cache alignment
inline bool apu_is_host_cache_aligned(const void * ptr, size_t offset = 0) {
    return ((reinterpret_cast<uintptr_t>(ptr) + offset) % APU_HOST_CACHE_ALIGNMENT_BYTES) == 0;
}

// RAII wrapper for AMDGPU GEM buffer object allocated on /dev/dri/renderD*
class apu_gem_buffer {
public:
    apu_gem_buffer(int render_fd, size_t size_bytes, bool map_cpu = true);
    ~apu_gem_buffer();

    apu_gem_buffer(const apu_gem_buffer &) = delete;
    apu_gem_buffer & operator=(const apu_gem_buffer &) = delete;

    apu_gem_buffer(apu_gem_buffer && other) noexcept;
    apu_gem_buffer & operator=(apu_gem_buffer && other) noexcept;

    int      get_render_fd()  const { return render_fd_; }
    uint32_t get_gem_handle() const { return gem_handle_; }
    int      get_prime_fd()   const { return prime_fd_; }
    void *   get_cpu_ptr()    const { return cpu_ptr_; }
    size_t   get_size()       const { return size_; }

    // DMA_BUF_IOCTL_SYNC CPU cache bracketing
    bool begin_cpu_access(bool write = true);
    bool end_cpu_access(bool write = true);

    // DMA_BUF_IOCTL_EXPORT_SYNC_FILE / DMA_BUF_IOCTL_IMPORT_SYNC_FILE
    int  export_sync_file(bool write = true);
    bool import_sync_file(int sync_fd, bool write = true);

private:
    int      render_fd_  = -1;
    uint32_t gem_handle_ = 0;
    int      prime_fd_   = -1;
    void *   cpu_ptr_    = nullptr;
    size_t   size_       = 0;
    bool     is_mapped_  = false;
};

// RAII wrapper for AMDXDNA NPU imported buffer object on /dev/accel/accel*
class apu_xdna_buffer {
public:
    apu_xdna_buffer(int accel_fd, int prime_fd, size_t size_bytes);
    ~apu_xdna_buffer();

    apu_xdna_buffer(const apu_xdna_buffer &) = delete;
    apu_xdna_buffer & operator=(const apu_xdna_buffer &) = delete;

    apu_xdna_buffer(apu_xdna_buffer && other) noexcept;
    apu_xdna_buffer & operator=(apu_xdna_buffer && other) noexcept;

    int      get_accel_fd()    const { return accel_fd_; }
    uint32_t get_xdna_handle() const { return xdna_handle_; }
    uint64_t get_xdna_addr()   const { return xdna_addr_; }
    uint64_t get_map_offset()  const { return map_offset_; }
    size_t   get_size()        const { return size_; }

    bool is_tile_dma_aligned(size_t offset = 0) const {
        if (xdna_addr_ != static_cast<uint64_t>(~0UL) && xdna_addr_ != 0) {
            return apu_is_tile_dma_aligned(xdna_addr_, offset);
        }
        return apu_is_tile_dma_aligned(map_offset_, offset);
    }

private:
    int      accel_fd_    = -1;
    uint32_t xdna_handle_ = 0;
    uint64_t xdna_addr_   = 0;
    uint64_t map_offset_  = 0;
    uint64_t vaddr_       = 0;
    size_t   size_        = 0;
};

// RAII wrapper for Linux DRM Synchronization Objects (drm_syncobj)
class apu_drm_syncobj {
public:
    explicit apu_drm_syncobj(int drm_fd, bool create_signaled = false);
    ~apu_drm_syncobj();

    apu_drm_syncobj(const apu_drm_syncobj &) = delete;
    apu_drm_syncobj & operator=(const apu_drm_syncobj &) = delete;

    apu_drm_syncobj(apu_drm_syncobj && other) noexcept;
    apu_drm_syncobj & operator=(apu_drm_syncobj && other) noexcept;

    int      get_drm_fd() const { return drm_fd_; }
    uint32_t get_handle() const { return syncobj_handle_; }

    // Timeline synchronization
    bool signal_timeline(uint64_t point);
    bool wait_timeline(uint64_t point, int64_t timeout_nsec = 5000000000LL);

    // Sync file interop
    int  export_sync_file();
    bool import_sync_file(int sync_fd);

private:
    int      drm_fd_         = -1;
    uint32_t syncobj_handle_ = 0;
};

// RAII scoped CPU access guard
class apu_dma_buf_cpu_scope {
public:
    apu_dma_buf_cpu_scope(apu_gem_buffer & buf, bool write = true)
        : buf_(buf), write_(write) {
        buf_.begin_cpu_access(write_);
    }
    ~apu_dma_buf_cpu_scope() {
        buf_.end_cpu_access(write_);
    }
private:
    apu_gem_buffer & buf_;
    bool write_;
};

// Physical hardware bridge smoke test runner
bool apu_run_bridge_smoke_test(bool verbose, apu_bridge_telemetry & telemetry, std::string & out_log);
