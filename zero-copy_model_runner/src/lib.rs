// SPDX-License-Identifier: Apache-2.0
//! Heterogeneous APU Orchestrator Runtime
//!
//! High-performance, zero-copy inference orchestrator coordinating CPU, iGPU, and NPU
//! across modern AMD Ryzen AI APU processors.

pub mod backend;
pub mod container;
pub mod doctor;
pub mod engine;
pub mod ffi;
pub mod memory;
pub mod model_cli;
pub mod router;
pub mod synth_cli;
pub mod topology;
pub mod uapi;

pub use ffi::{
    apu_backend_allocate_shared_kv, apu_backend_dispatch_decode_step, apu_backend_dispatch_prefill,
    apu_backend_doctor, apu_backend_free, apu_backend_get_hyperparams, apu_backend_load_model,
    apu_backend_model, apu_backend_synth, ApuBackendContext, ApuModelHyperparams,
};

pub use backend::{open_or_mock, BackendError, DeviceBackend, DeviceType, MockDeviceBackend, PhysicalDeviceBackend};
pub use container::{ContainerError, ModelHyperparameters, Q4nxHeader, Q4nxModel, Q4NX_MAGIC};
pub use engine::{
    ApuDecodeEngine, ApuPrefillEngine, DecodeEngine, DecodeStepRequest, DecodeStepResult,
    DeterministicReferenceOracle, PrefillEngine, PrefillRequest, PrefillResult, RocmPrefillEngine,
    Sampler, SamplerConfig, SpeculativeDraftingEngine, XrtDecodeEngine,
};
pub use memory::{
    DmaBufHandle, GpuBufferObject, MemoryBridge, MemoryError, MemoryPort, NpuBufferObject, SharedBuffer,
    SyncDirection, XrtBoHandle,
};
pub use topology::{ApuTopologyGovernor, CoreMicroarchitecture, CpuCoreInfo, TopologyError, WorkerRole};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_governor_initialization() {
        let governor = ApuTopologyGovernor::probe_system().expect("Failed to probe system topology");
        assert!(!governor.zen5_classic_cores().is_empty());
    }
}
