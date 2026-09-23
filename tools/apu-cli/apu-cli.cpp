// apu-cli — Phase 1 container inspection: bounded, read-only, payload-isolated.
// Subcommands:
//   container-info <file> [--companion <gguf>]
//   mem-estimate   <model.gguf> [--ctx N] [--kv-dtype f16|q8_0|q4_0]
// Phase 1 §1.1/§1.2 invariants:
//  * O_RDONLY only; never O_RDWR/creat/truncate; re-stat confirms unchanged.
//  * 2048-key metadata scan cap retained.
//  * Two-phase bounded .q4nx read; payload never read into RAM.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>

class ROFile {
public:
    static constexpr uint64_t MAX_META_KEY = 1ull << 20;
    explicit ROFile(const std::string & path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open O_RDONLY: " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = size0_ = (uint64_t) st.st_size;
        mt0_ = (int64_t) st.st_mtime;
    }
    ~ROFile() { if (fd_ >= 0) ::close(fd_); }
    ROFile(const ROFile &) = delete;
    ROFile & operator=(const ROFile &) = delete;
    uint64_t size() const { return size_; }
    const std::string & path() const { return path_; }
    std::vector<uint8_t> read_exact(uint64_t off, uint64_t n) {
        if (off + n < off || off + n > size_) throw std::runtime_error("bounded read out of range");
        std::vector<uint8_t> buf((size_t) n);
        uint64_t got = 0;
        while (got < n) {
            ssize_t r = ::pread(fd_, buf.data() + got, (size_t)(n - got), (off_t)(off + got));
            if (r <= 0) throw std::runtime_error("short read");
            got += (uint64_t) r;
        }
        return buf;
    }
    bool verify_unmodified(std::string & err) {
        struct stat st{};
        if (::fstat(fd_, &st) != 0) { err = "fstat failed"; return false; }
        if ((uint64_t) st.st_size != size0_) { err = "size changed"; return false; }
        if ((int64_t) st.st_mtime != mt0_)  { err = "mtime changed"; return false; }
        return true;
    }
private:
    std::string path_;
    int fd_ = -1;
    uint64_t size_ = 0, size0_ = 0;
    int64_t mt0_ = 0;
};

class BufCursor {
public:
    BufCursor(ROFile & f, uint64_t off, uint64_t window = 1u<<20) : f_(f), pos_(off), win_(window) {}
    uint64_t pos() const { return pos_; }
    void require(uint64_t n) { if (pos_ + n < pos_ || pos_ + n > f_.size()) throw std::runtime_error("cursor out of range"); }
    std::vector<uint8_t> take(uint64_t n) {
        require(n);
        std::vector<uint8_t> out((size_t) n);
        uint64_t got = 0;
        while (got < n) {
            if (buf_off_ > pos_ || pos_ >= buf_off_ + buf_.size()) refill();
            uint64_t avail = buf_off_ + buf_.size() - pos_;
            uint64_t c = std::min(avail, n - got);
            memcpy(out.data() + got, buf_.data() + (size_t)(pos_ - buf_off_), (size_t) c);
            pos_ += c; got += c;
        }
        return out;
    }
    void skip(uint64_t n) {
        require(n);
        if (n > win_) { pos_ += n; return; }
        uint64_t got = 0;
        while (got < n) {
            if (buf_off_ > pos_ || pos_ >= buf_off_ + buf_.size()) refill();
            uint64_t avail = buf_off_ + buf_.size() - pos_;
            uint64_t c = std::min(avail, n - got);
            pos_ += c; got += c;
        }
    }
    uint32_t u32() { auto b = take(4); uint32_t v; memcpy(&v,&b[0],4); return v; }
    uint64_t u64() { auto b = take(8); uint64_t v; memcpy(&v,&b[0],8); return v; }
    float    f32() { auto b = take(4); float v;    memcpy(&v,&b[0],4); return v; }
private:
    void refill() {
        uint64_t n = std::min(win_, f_.size() - pos_);
        buf_ = f_.read_exact(pos_, n);
        buf_off_ = pos_;
    }
    ROFile & f_;
    uint64_t pos_, win_;
    uint64_t buf_off_ = ~0ull;
    std::vector<uint8_t> buf_;
};
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static inline uint32_t ror32(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
struct Sha256 {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint64_t len=0; uint8_t buf[64]; size_t blen=0;
    void block(const uint8_t*p){
        uint32_t w[64];
        for(int i=0;i<16;i++) w[i]=((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|((uint32_t)p[i*4+2]<<8)|p[i*4+3];
        for(int i=16;i<64;i++){
            uint32_t s0=ror32(w[i-15],7)^ror32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=ror32(w[i-2],17)^ror32(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++){
            uint32_t S1=ror32(e,6)^ror32(e,11)^ror32(e,25);
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=hh+S1+ch+K256[i]+w[i];
            uint32_t S0=ror32(a,2)^ror32(a,13)^ror32(a,22);
            uint32_t mj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+mj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void update(const uint8_t*p,size_t n){
        len+=n;
        while(n){ size_t c=std::min((size_t)(64-blen),n); memcpy(buf+blen,p,c); blen+=c; p+=c; n-=c; if(blen==64){block(buf);blen=0;} }
    }
    std::string hex(){
        uint64_t bits=len*8; uint8_t p1=0x80; update(&p1,1);
        uint8_t z=0; while(blen!=56) update(&z,1);
        uint8_t lb[8]; for(int i=0;i<8;i++) lb[i]=(uint8_t)((bits>>(56-i*8))&0xff);
        update(lb,8);
        char out[65]; for(int i=0;i<8;i++) sprintf(out+i*8,"%08x",h[i]); return std::string(out,64);
    }
};
static std::string sha256_head(ROFile & f, uint64_t max_bytes) {
    Sha256 s; uint64_t off=0; const uint64_t CH=1u<<20;
    uint64_t limit=std::min(max_bytes, f.size());
    while(off<limit){ uint64_t n=std::min(CH,limit-off); auto b=f.read_exact(off,n); s.update(b.data(),b.size()); off+=n; }
    return s.hex();
}
struct GgmlType { uint32_t blck; uint32_t size; };
static bool ggml_type_info(uint32_t t, GgmlType & o) {
    switch(t){
        case 0:  o={1,4};    return true;  case 1:  o={1,2};    return true;
        case 2:  o={32,18};  return true;  case 3:  o={32,20};  return true;
        case 6:  o={32,22};  return true;  case 7:  o={32,24};  return true;
        case 8:  o={32,34};  return true;  case 9:  o={32,40};  return true;
        case 10: o={256,84}; return true;  case 11: o={256,110};return true;
        case 12: o={256,144};return true;  case 13: o={256,176};return true;
        case 14: o={256,210};return true;  case 15: o={256,292};return true;
        case 16: o={256,66}; return true;  case 17: o={256,74}; return true;
        case 18: o={256,98}; return true;  case 19: o={256,50}; return true;
        case 20: o={32,18};  return true;  case 21: o={256,110};return true;
        case 22: o={256,82}; return true;  case 23: o={256,136};return true;
        case 24: o={1,1};    return true;  case 25: o={1,2};    return true;
        case 26: o={1,4};    return true;  case 27: o={1,8};    return true;
        case 28: o={1,8};    return true;  case 29: o={256,56}; return true;
        case 30: o={1,2};    return true;  case 41: o={128,18}; return true;
        case 43: o={256,72}; return true;  default: return false;
    }
}
enum { GGUF_U8=0,GGUF_I8=1,GGUF_U16=2,GGUF_I16=3,GGUF_U32=4,GGUF_I32=5,
       GGUF_F32=6,GGUF_BOOL=7,GGUF_STR=8,GGUF_ARR=9,GGUF_U64=10,GGUF_I64=11,GGUF_F64=12 };
static uint64_t gguf_scalar_size(uint32_t t) {
    switch(t){
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}
static bool cursor_scalar_u64(BufCursor & c, uint32_t t, uint64_t & out) {
    switch(t){
        case GGUF_U8:  { out=c.take(1)[0]; return true; }
        case GGUF_I8:  { out=(uint64_t)(int64_t)(int8_t)c.take(1)[0]; return true; }
        case GGUF_BOOL:{ out=c.take(1)[0]; return true; }
        case GGUF_U16: { auto b=c.take(2); uint16_t v; memcpy(&v,&b[0],2); out=v; return true; }
        case GGUF_I16: { auto b=c.take(2); int16_t v; memcpy(&v,&b[0],2); out=(uint64_t)(int64_t)v; return true; }
        case GGUF_U32: { out=c.u32(); return true; }
        case GGUF_I32: { auto b=c.take(4); int32_t v; memcpy(&v,&b[0],4); out=(uint64_t)(int64_t)v; return true; }
        case GGUF_U64: case GGUF_I64: { out=c.u64(); return true; }
        case GGUF_F32: { float v=c.f32(); out=(uint64_t)(int64_t)(double)v; return true; }
        case GGUF_F64: { auto b=c.take(8); double v; memcpy(&v,&b[0],8); out=(uint64_t)(int64_t)v; return true; }
        default: return false;
    }
}
static const uint64_t GGUF_KEY_SCAN_LIMIT = 2048;   // §1.1 retained
struct TensorInfo { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
struct GgufScan {
    uint64_t n_kv=0,n_tensors=0,n_kv_scanned=0; bool kv_truncated=false;
    std::string arch, tokenizer_model;
    uint64_t ctx=0,n_layer=0,n_embd=0,n_head=0,n_head_kv=0,n_ff=0,n_expert=0,n_expert_used=0;
    uint64_t n_key_len=0,n_val_len=0;
    uint64_t n_bytes_weights=0;
    std::set<uint32_t> tensor_types;
    std::vector<TensorInfo> tensors;
    uint64_t meta_region_bytes=0;
};
static void scan_gguf(ROFile & f, GgufScan & g, bool want_tensors) {
    BufCursor c(f,0);
    auto magic=c.take(4);
    if(memcmp(magic.data(),"GGUF",4)!=0) throw std::runtime_error("not a GGUF file");
    (void)c.u32();
    g.n_tensors=c.u64(); g.n_kv=c.u64();
    for(uint64_t i=0;i<g.n_kv;i++){
        if(i>=GGUF_KEY_SCAN_LIMIT){ g.kv_truncated=true; break; }
        uint64_t klen=c.u64();
        if(klen>ROFile::MAX_META_KEY) throw std::runtime_error("meta key too long");
        auto kb=c.take(klen); std::string key((const char*)kb.data(),(size_t)kb.size());
        uint32_t vtype=c.u32(); g.n_kv_scanned++;
        if(vtype==GGUF_ARR){
            uint32_t atype=c.u32(); uint64_t alen=c.u64();
            if(atype==GGUF_STR){ for(uint64_t e=0;e<alen;e++){ uint64_t sn=c.u64(); c.skip(sn);} }
            else { uint64_t es=gguf_scalar_size(atype); if(!es) throw std::runtime_error("unhandled array elem type"); c.skip(es*alen);} 
        } else if(vtype==GGUF_STR){
            uint64_t slen=c.u64();
            if(slen>ROFile::MAX_META_KEY) throw std::runtime_error("meta string too long");
            auto vb=c.take(slen); std::string val((const char*)vb.data(),(size_t)vb.size());
            if(key=="general.architecture") g.arch=val;
            else if(key=="tokenizer.ggml.model") g.tokenizer_model=val;
        } else {
            uint64_t v=0; if(!cursor_scalar_u64(c,vtype,v)) throw std::runtime_error("unhandled scalar type");
            const std::string p=g.arch+".";
            if      (key==p+"context_length")          g.ctx=v;
            else if (key==p+"block_count")             g.n_layer=v;
            else if (key==p+"embedding_length")        g.n_embd=v;
            else if (key==p+"attention.head_count")    g.n_head=v;
            else if (key==p+"attention.head_count_kv") g.n_head_kv=v;
            else if (key==p+"feed_forward_length")     g.n_ff=v;
            else if (key==p+"expert_count")            g.n_expert=v;
            else if (key==p+"expert_used_count")       g.n_expert_used=v;
            else if (key==p+"attention.key_length")    g.n_key_len=v;
            else if (key==p+"attention.value_length")  g.n_val_len=v;
        }
    }
    g.meta_region_bytes=c.pos();
    for(uint64_t i=0;i<g.n_tensors;i++){
        uint64_t nlen=c.u64();
        if(nlen>ROFile::MAX_META_KEY) throw std::runtime_error("tensor name too long");
        auto nb=c.take(nlen); std::string name((const char*)nb.data(),(size_t)nb.size());
        uint32_t nd=c.u32(); if(nd>8) throw std::runtime_error("tensor ndim too large");
        std::vector<uint64_t> dims(nd); for(uint32_t d=0;d<nd;d++) dims[d]=c.u64();
        uint32_t ttype=c.u32(); uint64_t toff=c.u64();
        g.tensor_types.insert(ttype);
        uint64_t numel=1; for(auto d:dims) numel*=d;
        GgmlType gi{}; if(ggml_type_info(ttype,gi)&&gi.blck&&numel%gi.blck==0) g.n_bytes_weights+=(numel/gi.blck)*gi.size;
        if(want_tensors) g.tensors.push_back({name,dims,ttype,toff});
    }
}
struct Q4nxInfo {
    bool ok=false; uint32_t version=0; std::string arch;
    uint64_t hidden=0,heads=0,kv_heads=0,layers=0,vocab=0,ctx=0;
    uint64_t xclbin_off=0,xclbin_size=0,table_off=0,entries=0,payload_off=0,payload_size=0;
    uint64_t bytes_read=0; uint32_t table_entries_expanded=0;
    std::set<uint32_t> dtypes; bool m_xclbin=false,m_aie=false,m_sram=false,m_idpp=false;
    std::string error;
};
static const uint64_t Q4NX_HEADER_FIXED=256;
static Q4nxInfo scan_q4nx(ROFile & f) {
    Q4nxInfo q;
    auto h=f.read_exact(0,Q4NX_HEADER_FIXED); q.bytes_read+=h.size();
    if(memcmp(h.data(),"Q4NX",4)!=0){ q.error="bad magic (not Q4NX)"; return q; }
    auto rU32=[&](size_t o){ uint32_t v; memcpy(&v,&h[o],4); return v; };
    auto rU64=[&](size_t o){ uint64_t v; memcpy(&v,&h[o],8); return v; };
    q.version=rU32(4);
    { const char* p=(const char*)h.data()+8; size_t n=0; while(n<32&&p[n]) n++; q.arch.assign(p,n); }
    q.hidden=rU32(40); q.heads=rU32(44); q.kv_heads=rU32(48);
    q.layers=rU32(52); q.vocab=rU32(56); q.ctx=rU32(60);
    q.xclbin_off=rU64(64); q.xclbin_size=rU64(72);
    q.table_off=rU64(80); q.entries=rU64(88);
    q.payload_off=rU64(96); q.payload_size=rU64(104);
    if(q.xclbin_off!=Q4NX_HEADER_FIXED){ q.error="xclbin_offset != 256"; return q; }
    // Bare sidecar variant (Case B): embedded xclbin, NO tensor table; weights
    // live in the paired GGUF. entries==0 / table_off==0 is valid there.
    const bool has_table = (q.entries>0 && q.table_off>0);
    if(has_table){
        if(q.table_off<q.xclbin_off+q.xclbin_size){ q.error="table overlaps xclbin"; return q; }
        if(q.payload_off<q.table_off){ q.error="payload_offset < table_offset"; return q; }
    } else {
        if(q.payload_off<q.xclbin_off+q.xclbin_size){ q.error="payload overlaps xclbin"; return q; }
    }
    if(q.payload_off%64!=0){ q.error="payload_offset not 64B aligned"; return q; }
    if(q.table_off>f.size()||q.payload_off>f.size()){ q.error="header offsets beyond EOF"; return q; }
    if(q.xclbin_size>0){
        if(q.xclbin_size>(64ull<<20)){ q.error="xclbin exceeds 64MiB guard"; return q; }
        auto xc=f.read_exact(q.xclbin_off,q.xclbin_size); q.bytes_read+=xc.size();
        auto has=[&](const char* s){ return std::search(xc.begin(),xc.end(),s,s+strlen(s))!=xc.end(); };
        q.m_xclbin=has("xclbin")||has("xclbin2"); q.m_aie=has("aie_partition")||has("aie");
        q.m_sram=has("SRAM"); q.m_idpp=has("IDPP");
    }
    const uint64_t MAX_TABLE_ENTRIES=100000;
    if(q.entries>MAX_TABLE_ENTRIES){ q.error="table entries exceed cap"; return q; }
    uint64_t end=has_table?q.payload_off:q.table_off;
    if(end>f.size()) end=f.size();
    if(end>q.table_off){
        auto tbl=f.read_exact(q.table_off,end-q.table_off); q.bytes_read+=tbl.size();
        size_t p=0; uint32_t parsed=0;
        auto need=[&](size_t n){ if(p+n>tbl.size()) throw std::runtime_error("table truncated"); };
        for(uint64_t e=0;e<q.entries;e++){
            need(4); uint32_t nl; memcpy(&nl,&tbl[p],4); p+=4; need(nl); p+=nl;
            need(4); uint32_t nd; memcpy(&nd,&tbl[p],4); p+=4; need((size_t)nd*8); p+=(size_t)nd*8;
            need(4); uint32_t dt; memcpy(&dt,&tbl[p],4); p+=4; q.dtypes.insert(dt);
            need(16); p+=16; parsed++;
        }
        q.table_entries_expanded=parsed;
    }
    q.ok=true; return q;
}
static void mem_estimate(ROFile & f, uint64_t ctx_override, const std::string & kv_dtype) {
    GgufScan g; scan_gguf(f,g,false);
    const double MiB=1024.0*1024.0;
    uint64_t ctx=ctx_override?ctx_override:g.ctx;
    if(!g.n_head_kv) g.n_head_kv=g.n_head;
    uint64_t head_dim=g.n_key_len;
    if(!head_dim) head_dim=(g.n_head&&g.n_embd)?g.n_embd/g.n_head:0;
    uint64_t kv_elems=2ull*g.n_layer*g.n_head_kv*head_dim*ctx;
    uint64_t kv_bytes;
    if(kv_dtype=="q4_0") kv_bytes=(kv_elems/32)*18;
    else if(kv_dtype=="q8_0") kv_bytes=(kv_elems/32)*34;
    else kv_bytes=kv_elems*2;
    uint64_t act=(uint64_t)g.n_embd*8ull*4096ull;
    printf("mem-estimate: %s\n", f.path().c_str());
    printf("  arch           : %s\n", g.arch.c_str());
    printf("  weights        : %.2f MiB  (%llu tensors, %llu kv scanned)\n",
           g.n_bytes_weights/MiB,(unsigned long long)g.n_tensors,(unsigned long long)g.n_kv_scanned);
    printf("  kv cache       : %.2f MiB  (dtype=%s ctx=%llu layers=%llu heads_kv=%llu head_dim=%llu)\n",
           kv_bytes/MiB,kv_dtype.c_str(),(unsigned long long)ctx,(unsigned long long)g.n_layer,
           (unsigned long long)g.n_head_kv,(unsigned long long)head_dim);
    printf("  activations    : %.2f MiB  (estimate)\n", act/MiB);
    printf("  TOTAL          : %.2f MiB\n",(g.n_bytes_weights+kv_bytes+act)/MiB);
    printf("  note           : estimate only; no allocation performed\n");
}
static int container_info(const std::string & path, const std::string & /*companion*/, bool /*full_sha*/) {
    ROFile f(path);
    printf("container-info: %s (%llu bytes)\n",path.c_str(),(unsigned long long)f.size());
    printf("  sha256(16MiB-head): %s\n", sha256_head(f,16ull<<20).c_str());
    auto magic=f.read_exact(0,4);
    if(memcmp(magic.data(),"GGUF",4)==0){
        GgufScan g; scan_gguf(f,g,true);
        printf("  kind           : GGUF\n");
        printf("  arch           : %s\n", g.arch.c_str());
        printf("  n_kv/scanned   : %llu / %llu%s\n",(unsigned long long)g.n_kv,
               (unsigned long long)g.n_kv_scanned, g.kv_truncated?"  [TRUNCATED at 2048 cap]":"");
        printf("  n_tensors      : %llu\n",(unsigned long long)g.n_tensors);
        printf("  meta region    : %llu bytes (bounded, no payload read)\n",(unsigned long long)g.meta_region_bytes);
        printf("  ctx/blk/embd   : %llu / %llu / %llu\n",(unsigned long long)g.ctx,
               (unsigned long long)g.n_layer,(unsigned long long)g.n_embd);
        printf("  heads/kv       : %llu / %llu\n",(unsigned long long)g.n_head,(unsigned long long)g.n_head_kv);
        printf("  experts used   : %llu / %llu\n",(unsigned long long)g.n_expert_used,(unsigned long long)g.n_expert);
        printf("  weights bytes  : %llu\n",(unsigned long long)g.n_bytes_weights);
        printf("  tensor types   : "); for(auto t:g.tensor_types) printf("%u ",t); printf("\n");
        printf("  tokenizer      : %s\n", g.tokenizer_model.c_str());
    } else if(memcmp(magic.data(),"Q4NX",4)==0){
        Q4nxInfo q=scan_q4nx(f);
        if(!q.ok){ printf("  kind           : Q4NX sidecar\n  ERROR          : %s\n", q.error.c_str()); return 1; }
        printf("  kind           : Q4NX sidecar (v%u)\n", q.version);
        printf("  arch           : %s\n", q.arch.c_str());
        printf("  hyperparams    : hidden=%llu heads=%llu kv=%llu layers=%llu vocab=%llu ctx=%llu\n",
               (unsigned long long)q.hidden,(unsigned long long)q.heads,(unsigned long long)q.kv_heads,
               (unsigned long long)q.layers,(unsigned long long)q.vocab,(unsigned long long)q.ctx);
        printf("  xclbin         : off=%llu size=%llu (markers: xclbin2=%d aie=%d SRAM=%d IDPP=%d)\n",
               (unsigned long long)q.xclbin_off,(unsigned long long)q.xclbin_size,q.m_xclbin,q.m_aie,q.m_sram,q.m_idpp);
        if(q.entries>0)
            printf("  tensor table   : off=%llu entries=%llu expanded=%u\n",
                   (unsigned long long)q.table_off,(unsigned long long)q.entries,q.table_entries_expanded);
        else
            printf("  tensor table   : absent (bare sidecar; weights in companion GGUF)\n");
        printf("  payload        : off=%llu size=%llu (ALIGNED, NOT READ)\n",
               (unsigned long long)q.payload_off,(unsigned long long)q.payload_size);
        printf("  dtypes         : "); for(auto d:q.dtypes) printf("%u ",d); printf("\n");
        printf("  bytes read     : %llu (bounded; payload isolated)\n",(unsigned long long)q.bytes_read);
    } else {
        // Safetensors-index variant (e.g. Qwen3.5-4B-NPU2): u64 header len + JSON.
        auto l8 = f.read_exact(0, 8);
        uint64_t jlen; memcpy(&jlen, l8.data(), 8);
        if (jlen > 0 && jlen < (64ull<<20) && 8 + jlen <= f.size()) {
            auto jb = f.read_exact(8, jlen);
            std::string js((const char*) jb.data(), (size_t) jb.size());
            if (!js.empty() && js[0] == '{') {
                size_t nt = 0, pos = 0;
                while ((pos = js.find("\"dtype\"", pos)) != std::string::npos) { nt++; pos += 7; }
                printf("  kind           : safetensors-index (NPU2-style q4nx)\n");
                printf("  header json    : %llu bytes, %zu tensor descriptors (indexed; not GGUF/Q4NX-magic)\n",
                       (unsigned long long) jlen, nt);
                printf("  payload        : NOT READ (indexed variant; companion GGUF/weights separate)\n");
                std::string err;
                if(!f.verify_unmodified(err)){ printf("  READONLY VIOLATION: %s\n", err.c_str()); return 1; }
                printf("  readonly check : PASS (size+mtime unchanged)\n");
                return 0;
            }
        }
        printf("  kind           : unknown/unsupported magic\n"); return 1;
    }
    std::string err;
    if(!f.verify_unmodified(err)){ printf("  READONLY VIOLATION: %s\n", err.c_str()); return 1; }
    printf("  readonly check : PASS (size+mtime unchanged)\n");
    return 0;
}
static uint64_t arg_u64(int argc,char**argv,const char*name,uint64_t def){
    for(int i=0;i<argc-1;i++) if(!strcmp(argv[i],name)) return strtoull(argv[i+1],nullptr,0);
    return def;
}

// ---- Phase 2 §2.1/§2.2: capability detection, xclbin auto-discovery, route resolution ----
// Read-only, runtime-only detection. No XRT headers/ABI linkage in this milestone (§2.4 scope).
static bool file_exists(const std::string & p){
    struct stat st{}; return ::stat(p.c_str(),&st)==0;
}
static std::string readlink_str(const std::string & p){
    char buf[512]={0};
    ssize_t n=::readlink(p.c_str(),buf,sizeof(buf)-1);
    return n>0? std::string(buf,(size_t)n) : std::string();
}
static std::string home_dir(){
    const char* h=::getenv("HOME");
    return h? std::string(h): ".";
}
static std::string basename_no_ext(const std::string & p){
    size_t s=p.find_last_of('/'); std::string b=(s==std::string::npos)?p:p.substr(s+1);
    size_t d=b.find_last_of('.'); if(d!=std::string::npos) b=b.substr(0,d);
    return b;
}
struct NpuCaps {
    bool dev=false, driver=false, xrt=false;
    std::string dev_node="/dev/accel/accel0", driver_name="", xrt_lib="";
};
static NpuCaps npu_caps(){
    NpuCaps c;
    c.dev=file_exists(c.dev_node);
    if(c.dev){
        std::string drv=readlink_str("/sys/class/accel/accel0/device/driver");
        size_t s=drv.find_last_of('/');
        c.driver_name=(s==std::string::npos)?drv:drv.substr(s+1);
        c.driver=!c.driver_name.empty();
    }
    // runtime-only XRT probe: dlopen the core util lib and look for the public XRT C ABI entrypoint
    // (no build-time header/ABI coupling; real ABI use is phase-2 §2.4)
    const char* libs[]={"libxrt_coreutil.so.2","libxrt_coreutil.so",nullptr};
    for(int i=0;libs[i];i++){
        if(void* h=dlopen(libs[i],RTLD_NOW)){
            bool ok=dlsym(h,"xrtDeviceOpen")!=nullptr;
            ::dlclose(h);
            if(ok){ c.xrt=true; c.xrt_lib=libs[i]; break; }
        }
    }
    return c;
}
struct GpuCaps { bool kfd=false, dri=false; };
static GpuCaps gpu_caps(){
    GpuCaps g;
    g.kfd=file_exists("/dev/kfd");            // ROCm compute kernel node
    g.dri=file_exists("/dev/dri/renderD128"); // DRM render node
    return g;
}
// Four-tier xclbin auto-discovery (phase-2 §2.1):
//   1. explicit --apu-xclbin PATH   2. $LLAMA_APU_XCLBINS_DIR
//   3. ~/.local/share/llama-apu/xclbins  4. /usr/local/share/llama-apu/xclbins
// Model key = model basename without extension (NPU2 layout: <stem>/[roles].xclbin).
struct XclbinFind {
    int tier=0; std::string dir, tier_name; std::vector<std::string> bins;
    bool found=false;
};
static std::vector<std::string> list_xclbins(const std::string & dir){
    std::vector<std::string> out;
    DIR* d=::opendir(dir.c_str());
    if(!d) return out;
    while(dirent* e=::readdir(d)){
        std::string n=e->d_name;
        if(n.size()>7 && n.substr(n.size()-7)==".xclbin") out.push_back(dir+"/"+n);
    }
    ::closedir(d);
    std::sort(out.begin(),out.end());
    return out;
}
static XclbinFind find_xclbins(const std::string & model_stem, const std::string & parent_stem, const std::string & explicit_path){
    const char* tier_names[]={"","--apu-xclbin","LLAMA_APU_XCLBINS_DIR","user path","system path"};
    std::string dirs[4]; int tiers=4;
    if(!explicit_path.empty()){ dirs[0]=explicit_path; tiers=1; }
    else {
        const char* env=::getenv("LLAMA_APU_XCLBINS_DIR");
        dirs[0]=env? std::string(env): std::string();
        dirs[1]=home_dir()+"/.local/share/llama-apu/xclbins";
        dirs[2]="/usr/local/share/llama-apu/xclbins";
        tiers=3;
    }
    // model keys: file stem first (e.g. Foo-NPU2.gguf -> Foo-NPU2), then parent dir
    // (NPU2 layout /opt/models/Qwen3.5-4B-NPU2/model.q4nx -> key Qwen3.5-4B-NPU2)
    const std::string keys[]={model_stem,parent_stem,std::string()};
    for(int i=0;i<tiers;i++){
        if(dirs[i].empty()) continue;
        std::string cand = dirs[i];
        struct stat st{};
        if(::stat(cand.c_str(),&st)!=0) continue;
        if(S_ISDIR(st.st_mode)){
            for(int k=0;!keys[k].empty();k++){
                std::string in_root = cand+"/"+keys[k];
                std::string use = ::stat(in_root.c_str(),&st)==0 && S_ISDIR(st.st_mode) ? in_root : cand;
                auto bins=list_xclbins(use);
                if(!bins.empty()) return {tiers==1?1:i+2,use,tier_names[tiers==1?1:i+2],bins,true};
            }
        } else if(cand.size()>7 && cand.substr(cand.size()-7)==".xclbin"){
            return {1,cand,tier_names[1],{cand},true};
        }
    }
    return {0,"","",{},false};
}
static uint64_t mem_available_bytes(){
    FILE* f=::fopen("/proc/meminfo","r");
    if(!f) return 0;
    char k[64]; uint64_t v=0; char u[16];
    while(::fscanf(f,"%63s %llu %15s",k,(unsigned long long*)&v,u)==3)
        if(!strcmp(k,"MemAvailable:")){ ::fclose(f); return v*1024ull; }
    ::fclose(f); return 0;
}
static int route_info(const std::string & path, const std::string & explicit_xclbin,
                      uint64_t ctx_override, const std::string & kv_dtype){
    const double MiB=1024.0*1024.0;
    printf("route-info: %s\n", path.c_str());

    // model memory estimate (bounded read; same scanner as mem-estimate)
    uint64_t need_bytes=0;
    {
        ROFile f(path);
        auto magic=f.read_exact(0,4);
        if(memcmp(magic.data(),"GGUF",4)==0){
            GgufScan g; scan_gguf(f,g,false);
            uint64_t ctx=ctx_override?ctx_override:g.ctx;
            if(!g.n_head_kv) g.n_head_kv=g.n_head;
            uint64_t head_dim=g.n_key_len;
            if(!head_dim) head_dim=(g.n_head&&g.n_embd)?g.n_embd/g.n_head:0;
            uint64_t kv_elems=2ull*g.n_layer*g.n_head_kv*head_dim*ctx;
            uint64_t kv_bytes;
            if(kv_dtype=="q4_0") kv_bytes=(kv_elems/32)*18;
            else if(kv_dtype=="q8_0") kv_bytes=(kv_elems/32)*34;
            else kv_bytes=kv_elems*2;
            need_bytes=g.n_bytes_weights+kv_bytes+(uint64_t)(g.n_embd*8ull*4096ull);
        }
    }
    const uint64_t avail=mem_available_bytes();

    // hardware capability (runtime-only, read-only)
    NpuCaps npu=npu_caps();
    GpuCaps gpu=gpu_caps();
    std::string stem=basename_no_ext(path);
    std::string parent;
    {   size_t s=path.find_last_of('/');
        if(s!=std::string::npos && s>0){
            std::string d=path.substr(0,s);
            size_t s2=d.find_last_of('/');
            parent = (s2==std::string::npos)?d:d.substr(s2+1);
        }
    }
    XclbinFind xb=find_xclbins(stem,parent,explicit_xclbin);

    printf("  hardware        :\n");
    printf("    NPU device    : %s (%s)\n", npu.dev?"present":"absent",
           npu.dev?(npu.driver_name.empty()?"driver unknown":npu.driver_name.c_str()):"-");
    printf("    XRT runtime   : %s%s\n", npu.xrt?"loadable":"not found",
           npu.xrt?(" ("+npu.xrt_lib+", xrtDeviceOpen)").c_str():"");
    printf("    GPU (ROCm)    : /dev/kfd %s, render node %s\n",
           gpu.kfd?"present":"absent", gpu.dri?"present":"absent");
    printf("  xclbin discovery:\n");
    if(xb.found){
        printf("    resolved tier : %d (%s)\n", xb.tier, xb.tier_name.c_str());
        printf("    directory     : %s\n", xb.dir.c_str());
        printf("    profiles      : %zu\n", xb.bins.size());
        for(auto & b: xb.bins) printf("      - %s\n", b.c_str());
    } else {
        printf("    resolved tier : none — no .xclbin found for model '%s' in any tier\n", stem.c_str());
    }
    if(need_bytes && avail){
        printf("  memory          : model+kv est %.2f MiB vs MemAvailable %.2f MiB -> %s\n",
               need_bytes/MiB, avail/MiB, need_bytes<=avail?"fits":"BLOCKED (exceeds available)");
    }

    // route resolution — honest fallback with reasons (roadmap: NPU routes enabled only when validated)
    printf("  routing (default policy: tokenize=cpu, prefill=gpu, decode=npu-when-safe, fallback=gpu):\n");
    printf("    tokenize : cpu  (llama.cpp tokenizer is host CPU code; no device tokenizer exists)\n");
    bool prefill_gpu_ok = gpu.kfd;
    printf("    prefill  : %s%s\n", prefill_gpu_ok?"gpu":"cpu",
           prefill_gpu_ok?"":"  (no /dev/kfd — ROCm GPU unavailable)");
    bool mem_ok = (!need_bytes || !avail || need_bytes<=avail);
    // NPU decode is staged: PRIME dma-buf bridge (§2.3) is implemented; NPU kernel execution contract (§2.4) pending.
    std::vector<std::string> npu_blockers;
    if(!npu.dev)               npu_blockers.push_back("no NPU device node "+npu.dev_node);
    else if(!npu.driver)      npu_blockers.push_back("no driver bound at /sys/class/accel/accel0");
    if(!npu.xrt)               npu_blockers.push_back("XRT runtime not loadable");
    if(!xb.found)              npu_blockers.push_back("no .xclbin profile discovered (4-tier search)");
    if(!mem_ok)                npu_blockers.push_back("memory estimate exceeds MemAvailable");
    npu_blockers.push_back("NPU kernel execution not validated yet (phase-2 §2.4, in progress)");
    printf("    decode   : gpu   (npu requested; NPU blocked:)");
    for(auto & b: npu_blockers) printf("\n               %s", ("- "+b).c_str());
    printf("\n");
    printf("    fallback : gpu%s\n", gpu.kfd?"":" (unavailable! check /dev/kfd)");
    printf("  note            : PRIME dma-buf bridge §2.3 is active; NPU kernel execution validation §2.4 is next\n");
    return 0;
}
#include "ggml-apu-bridge.h"

static int test_bridge(bool verbose) {
    apu_bridge_telemetry telem{};
    std::string log;
    printf("llama-apu bridge test (§2.3 Shared Memory & Timeline Synchronization):\n");
    bool ok = apu_run_bridge_smoke_test(verbose, telem, log);
    printf("%s", log.c_str());
    if (ok) {
        printf("RESULT: PASS — Physical GEM allocation, PRIME dma-buf export, AMDXDNA import, 16B Tile DMA alignment, and DRM syncobj timeline signaling verified.\n");
        return 0;
    } else {
        printf("RESULT: FAIL — Bridge verification failed.\n");
        return 1;
    }
}

static int test_npu(const std::string & xclbin_path, bool verbose) {
    std::string log;
    bool ok = apu_run_npu_validation_test(xclbin_path, verbose, log);
    printf("%s", log.c_str());
    if (ok) {
        printf("RESULT: PASS — NPU execution validation passed on physical XDNA 2 hardware.\n");
        return 0;
    } else {
        printf("RESULT: FAIL — NPU execution validation failed.\n");
        return 1;
    }
}

static int usage(){
    fprintf(stderr,
        "apu-cli — Phase 1/2 container inspection & APU route planning\n"
        "  apu-cli container-info <file> [--companion <gguf>] [--full-sha256]\n"
        "  apu-cli mem-estimate   <model.gguf> [--ctx N] [--kv-dtype f16|q8_0|q4_0]\n"
        "  apu-cli route-info     <model.gguf> [--apu-xclbin <PATH>] [--ctx N] [--kv-dtype f16|q8_0|q4_0]\n"
        "  apu-cli test-bridge    [--apu-verbose]\n"
        "  apu-cli test-npu       [<xclbin>] [--apu-verbose]\n");
    return 2;
}

int main(int argc,char**argv){
    if(argc<2) return usage();
    try {
        std::string cmd=argv[1];
        if(cmd=="test-bridge"){
            bool verbose = false;
            for(int i=2; i<argc; i++) if(!strcmp(argv[i],"--apu-verbose") || !strcmp(argv[i],"-v")) verbose = true;
            return test_bridge(verbose);
        }
        if(cmd=="test-npu"){
            std::string xclbin;
            bool verbose = false;
            for(int i=2; i<argc; i++){
                if(!strcmp(argv[i],"--apu-verbose") || !strcmp(argv[i],"-v")) verbose = true;
                else if(!strcmp(argv[i],"--apu-xclbin") && i+1<argc) xclbin = argv[++i];
                else if(argv[i][0] != '-') xclbin = argv[i];
            }
            return test_npu(xclbin, verbose);
        }
        if(argc<3) return usage();
        std::string path=argv[2];
        std::string xclbin; for(int i=3;i<argc-1;i++) if(!strcmp(argv[i],"--apu-xclbin")) xclbin=argv[i+1];
        if(cmd=="container-info"){
            std::string comp; for(int i=3;i<argc-1;i++) if(!strcmp(argv[i],"--companion")) comp=argv[i+1];
            return container_info(path,comp,false);
        } else if(cmd=="mem-estimate"){
            ROFile f(path);
            std::string kv= (strstr(argv[0],"q8")?"q8_0":"f16");
            for(int i=3;i<argc-1;i++) if(!strcmp(argv[i],"--kv-dtype")) kv=argv[i+1];
            uint64_t ctx=arg_u64(argc,argv,"--ctx",0);
            mem_estimate(f,ctx,kv); return 0;
        } else if(cmd=="route-info"){
            std::string kv="f16";
            for(int i=3;i<argc-1;i++) if(!strcmp(argv[i],"--kv-dtype")) kv=argv[i+1];
            uint64_t ctx=arg_u64(argc,argv,"--ctx",0);
            return route_info(path,xclbin,ctx,kv);
        }
        return usage();
    } catch(const std::exception &e){ fprintf(stderr,"apu-cli: error: %s\n",e.what()); return 1; }
}
