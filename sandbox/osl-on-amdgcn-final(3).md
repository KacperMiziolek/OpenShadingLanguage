# OSL on AMDGCN

This document consolidates the current AMDGCN proposal into one final, uniform design note. The goal is to add AMDGPU support to OSL without hard-coding a second backend-specific path beside OptiX. The first milestone is a backend-neutral GPU artifact pipeline that can compile a shader group for a selected GPU backend, emit backend artifacts, store them on the ShaderGroup, expose them through getattribute, save them from testshade, and make them available to a downstream runtime such as HIPRTC or an offline LLVM or LLD flow.

The first milestone does not attempt to replace the OptiX runtime path, design a universal renderer API for every GPU backend, solve batched GPU execution unless required, or promise cross-LLVM-version stability for serialized LLVM IR or bitcode. Artifact generation is intentionally separated from runtime execution so that OSL can validate the architecture before it owns a complete HIP execution stack.

## Current State

OSL currently has two materially different code generation modes: host JIT via MCJIT and OptiX/NVPTX-oriented device emission. The OptiX path is not a generic GPU abstraction. It is a specialized path threaded through backend selection, LLVM module setup, optimization, artifact storage, caching, CLI options, and renderer integration.

The current implementation is shaped around these facts:

- ShadingSystemImpl initializes GPU mode as m_use_optix(renderer->supports("OptiX")) and cache support from renderer->supports("optix_ptx_cache") in src/liboslexec/shadingsys.cpp.
- BackendLLVM caches that state as m_use_optix in src/liboslexec/backendllvm.cpp.
- llvm_instance.cpp loads OptiX-oriented seed bitcode, sets the NVPTX target triple and data layout explicitly, tags library functions for OptiX, and emits PTX into group().m_llvm_ptx_compiled_version.
- ShaderGroup stores only a PTX string for GPU output today.
- ShadingSystemImpl::getattribute(group, "ptx_compiled_version", ...) exposes only that PTX string.
- LLVM_Util exposes NVPTX-biased APIs such as nvptx_target_machine() and ptx_compile_group(...).
- testshade exposes --optix and --saveptx and hands PTX-oriented controls to the OptiX renderer.
- testrender is split between a CPU SimpleRaytracer and an OptixRaytracer; it is not organized around a generic GPU runtime interface.

OSL also already manages three distinct classes of device artifacts, and the AMD design has to account for all three:

1. Built-in OSL shadeops compiled for device use.
2. Renderer-supplied support-library code such as rend_lib.
3. Per-shader-group compiled output.

## Existing Bitcode and PTX Pipeline

The current pipeline is spread across multiple CMake helpers and runtime code paths.

### Host-side embedded LLVM bitcode

src/cmake/llvm_macros.cmake defines EMBED_LLVM_BITCODE_IN_CPP(...). That helper compiles listed .cpp sources to LLVM assembly with clang++ -S -emit-llvm, assembles them to .bc with llvm-as, links them into one combined .bc with llvm-link -internalize, serializes the final .bc file into generated C++ via src/build-scripts/serialize-bc.py, and then compiles that generated C++ into the target binary. At runtime, OSL loads those embedded byte arrays with module_from_bitcode(...).

Examples in tree:

- src/liboslexec/CMakeLists.txt embeds osl_llvm_compiled_ops and osl_llvm_compiled_rs_dependent_ops.
- src/testshade/CMakeLists.txt embeds testshade_llvm_compiled_rs.

### Device-oriented CUDA and NVPTX bitcode and PTX

src/cmake/cuda_macros.cmake adds a second pipeline for the OptiX path.

NVCC_COMPILE(...) runs nvcc -ptx on CUDA sources such as src/testrender/cuda/optix_raytracer.cu and src/testshade/cuda/optix_grid_renderer.cu. Those PTX files are installed and loaded by OptiX renderers at runtime.

MAKE_CUDA_BITCODE(...) uses clang++ --language=cuda --cuda-device-only --cuda-gpu-arch=... -S -emit-llvm to produce device LLVM assembly and then .bc. CUDA_SHADEOPS_COMPILE(...) then compiles multiple device-facing sources to .bc, links them into one linked_<prefix>.bc, optimizes that bitcode with opt, lowers it to PTX with llc --march=nvptx64 -mcpu=<arch>, and post-processes the PTX via process-ptx.py. The results are serialized into C++ blobs with MAKE_EMBEDDED_CPP(...).

Current examples:

- src/liboslexec/CMakeLists.txt builds shadeops_cuda_llvm_compiled_ops and shadeops_cuda_ptx_compiled_ops.
- src/testrender/CMakeLists.txt builds rend_lib_llvm_compiled_ops and rend_lib_testrender.ptx.
- src/testshade/CMakeLists.txt builds rend_lib_llvm_compiled_ops and rend_lib_testshade.ptx.

### Runtime usage today

At runtime, llvm_instance.cpp follows two different flows:

- Host JIT loads host-side embedded bitcode, optionally absorbs renderer free-function bitcode, and JITs with MCJIT.
- The OptiX path loads embedded CUDA shadeops bitcode, optionally links renderer rend_lib bitcode, forces NVPTX target triple and data layout, optimizes, strips non-inlined library bodies, emits PTX, and stores that PTX on the ShaderGroup.

OptixRaytracer then retrieves renderer program PTX from installed files, built-in shadeops PTX from ShadingSystem::getattribute("shadeops_cuda_ptx", ...), and shader-group PTX from ShadingSystem::getattribute(group, "ptx_compiled_version", ...).

## Core Design Problem

The main architectural issue is not emitting AMDGPU once. The problem is that OSL currently treats GPU code generation as a boolean: use_optix(). If AMD support is added by scattering if (use_optix()) ... else if (use_amdgpu()) ... across shadingsys.cpp, backendllvm, llvm_instance.cpp, llvm_util, testshade, testrender, and CMake, the code will work briefly and then become difficult to evolve.

The migration has two separate dimensions that should not be collapsed:

- Compile-time guards such as #if OSL_USE_OPTIX and a new #if OSL_ENABLE_AMDGPU.
- Runtime backend selection, which should move from a boolean to an explicit target descriptor.

## Target Design

### GPU target descriptor

Replace the OptiX boolean model with an explicit descriptor carried by ShadingSystemImpl, BackendLLVM, and artifact consumers.

```cpp
enum class GPUBackendKind {
    None,
    NVPTX,
    AMDGPU
};

enum class GPUArtifactKind {
    None,
    PTX,
    LLVMBitcode,
    LLVMIR,
    HSACO
};

struct GPUTargetDesc {
    GPUBackendKind backend = GPUBackendKind::None;
    GPUArtifactKind artifact = GPUArtifactKind::None;
    std::string triple;
    std::string cpu;
    std::string features;
    std::string data_layout;
    std::vector<std::string> archs;
    bool rdc = false;
    int code_obj_version = 5;
    bool enable_cache = false;
};
```

Recommended initial mapping:

- CPU JIT: backend=None, artifact=None.
- OptiX: backend=NVPTX, artifact=PTX, triple=nvptx64-nvidia-cuda, rdc=false, archs={sm_XX}.
- AMD first milestone: backend=AMDGPU, artifact=LLVMBitcode, triple=amdgcn-amd-amdhsa, rdc=true, archs={gfx1100,...}.

Example constructor:

```cpp
GPUTargetDesc make_amdgpu_target(const std::vector<std::string>& archs) {
    GPUTargetDesc t;
    t.backend = GPUBackendKind::AMDGPU;
    t.artifact = GPUArtifactKind::LLVMBitcode;
    t.triple = "amdgcn-amd-amdhsa";
    t.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64"
                    "-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128"
                    "-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
                    "-v48:64-v96:128-v192:256-v256:256-v512:512"
                    "-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9";
    t.rdc = true;
    t.code_obj_version = 5;
    t.archs = archs;
    return t;
}
```

This descriptor becomes the single source of truth for module setup, artifact emission, cache keys, and renderer lookup.

### Generic ShaderGroup artifact storage

Replace the PTX-only storage model with a generic artifact container.

```cpp
enum class GPUExportKind {
    Init,
    EntryLayer,
    FusedEntry
};

struct GPUExportedSymbol {
    GPUExportKind kind = GPUExportKind::EntryLayer;
    std::string layer_name;
    std::string symbol_name;
};

struct CompiledGPUArtifact {
    GPUBackendKind backend = GPUBackendKind::None;
    GPUArtifactKind artifact = GPUArtifactKind::None;
    std::string triple;
    std::string arch;
    std::string llvm_version;
    bool rdc = false;
    std::vector<GPUExportedSymbol> exports;
    std::vector<uint8_t> payload;
};

struct ShaderGroup {
    std::vector<CompiledGPUArtifact> m_compiled_gpu_artifacts;
};
```

Using std::vector<uint8_t> keeps PTX representable while also supporting binary-safe LLVM bitcode and future HSACO payloads.

Backward compatibility should be preserved by keeping ptx_compiled_version as an alias for the first NVPTX PTX artifact while new code uses the generic model.

### Public artifact query contract

The new API should not extend the current undocumented ptx_compiled_version pointer convention. The backend-neutral contract should use documented getattribute shapes:

- TypeDesc::INT for scalar metadata.
- TypeDesc::STRING for ustring-backed textual metadata.
- TypeDesc(TypeDesc::UINT8, N) for binary payload copies.

The recommended query model is an indexed copy protocol rather than returning raw pointers into ShaderGroup memory:

```cpp
int count = 0;
shadingsys->getattribute(group, "gpu_num_artifacts", TypeDesc::INT, &count);

for (int i = 0; i < count; ++i) {
    int backend = 0, kind = 0, size = 0, rdc = 0, num_exports = 0;
    char triple[256] = {}, arch[64] = {};

    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:backend", i),
                             TypeDesc::INT, &backend);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:kind", i),
                             TypeDesc::INT, &kind);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:triple", i),
                             TypeDesc::STRING, triple);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:arch", i),
                             TypeDesc::STRING, arch);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:rdc", i),
                             TypeDesc::INT, &rdc);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:num_exports", i),
                             TypeDesc::INT, &num_exports);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:size", i),
                             TypeDesc::INT, &size);

    std::vector<uint8_t> payload(size);
    shadingsys->getattribute(group, fmtformat("gpu_artifact:{}:data", i),
                             TypeDesc(TypeDesc::UINT8, size), payload.data());
}
```

Because entry point ABI matters to both OptiX and HIP, one entry_name field is not enough. Artifact metadata should record multiple exported symbols, including init, one or more entry layers, and the fused callable if present.

GroupData alignment should also become an explicit attribute beside llvm_groupdata_size:

```cpp
int gd_size = 0;
int gd_align = 0;
shadingsys->getattribute(group, "llvm_groupdata_size", TypeDesc::INT, &gd_size);
shadingsys->getattribute(group, "llvm_groupdata_alignment", TypeDesc::INT,
                         &gd_align);
```

### Backend-neutral LLVM_Util surface

LLVM_Util should stop exposing NVPTX as the only first-class device backend. The call surface should become descriptor-driven even if the internal implementation still dispatches to backend-specific helpers.

```cpp
struct GPUEmitDesc {
    GPUBackendKind backend = GPUBackendKind::None;
    GPUArtifactKind artifact = GPUArtifactKind::None;
    std::string triple;
    std::string cpu;
    std::string features;
    std::string data_layout;
    bool rdc = false;
};

llvm::TargetMachine* target_machine_for(const GPUEmitDesc& desc);
bool emit_gpu_artifact(const GPUEmitDesc& desc,
                       llvm::Module* module,
                       std::vector<uint8_t>& out);
```

The first implementation can still be split into emit_nvptx_ptx(...), emit_amdgpu_bitcode(...), and emit_amdgpu_ir(...). The important change is the API boundary.

### Separate generic GPU lowering from OptiX ABI rules

Every use_optix() branch should be reclassified as one of four kinds:

1. Generic GPU legality or optimization behavior.
2. NVPTX-specific lowering behavior.
3. OptiX ABI and runtime behavior.
4. Behavior also required for AMDGPU.

That classification matters because AMDGPU should inherit generic GPU behavior, should not inherit OptiX ABI rules, and should only inherit selected NVPTX-specific logic when it is truly target-neutral.

One existing bug should be fixed before the larger refactor: fused_function_name() in llvm_instance.cpp currently prefixes __direct_callable__ unconditionally. That prefix should remain OptiX-only.

```cpp
std::string fused_function_name(const ShaderGroup& group)
{
    int nlayers = group.nlayers();
    ShaderInstance* inst = group[nlayers - 1];
    bool is_nvptx = inst->shadingsys().m_gpu_target.backend
                    == GPUBackendKind::NVPTX;
    const char* prefix = is_nvptx ? "__direct_callable__" : "";
    return fmtformat("{}fused_{}_name_{}", prefix, group.name(),
                     inst->layername());
}
```

### Renderer capability model

RendererServices::supports() should stop using "OptiX" as the proxy for backend mode. Recommended capability strings are:

- GPU
- NVPTX
- AMDGPU
- gpu_ptx_cache
- gpu_artifact_cache
- gpu_device_memory
- gpu_renderer_bitcode

Keeping the string-based API is fine for compatibility, but OSL core code should consume backend-neutral capability names instead of using "OptiX" as the master switch.

## AMD Artifact Strategy

LLVM bitcode should be the primary AMD interchange artifact. PTX is the correct output for OptiX because OptiX consumes PTX. For AMDGPU, textual LLVM IR is useful for debugging, but LLVM bitcode is the better first-class transport and cache payload because it is binary-safe, smaller than textual IR, closer to downstream LLVM-based consumers, and easier for OSL to keep opaque.

Recommended artifact priorities:

- Primary artifact: linked AMDGPU-targeted LLVM bitcode.
- Optional debug artifact: textual LLVM IR written only when explicitly requested.
- Optional later runtime artifact: HSACO once the HIP execution path is mature enough to justify it.

The document should explicitly distinguish two renderer integration modes:

1. Monolithic emit-only mode, where OSL links shadeops bitcode, renderer support-library bitcode, and per-group IR into one arch-qualified module and returns that module as the artifact.
2. Split-link runtime mode, where OSL returns group-specific bitcode plus metadata and the renderer performs the final link.

The first milestone should implement monolithic emit-only mode. That matches the current OptiX architecture more closely because the renderer already injects one lib_bitcode blob and OSL links it before artifact emission.

The runtime boundary must also be explicit. Raw LLVM bitcode is a good OSL-internal transport artifact, but a HIP execution path will eventually need either offline-generated HSACO per target arch or a documented runtime compilation step from LLVM IR or bitcode to a code object. The intended staged runtime contract is:

```text
OSL shadeops .bc
 + renderer support-library .bc
 + shader-group .bc
 -> llvm-link
 -> verification and exported-symbol audit
 -> code-object generation
 -> HSACO loaded by HIP module APIs
```

### Multi-arch behavior

Multi-arch handling should be explicit from the start:

- Each gfx... value produces one distinct artifact.
- Each artifact records its own arch.
- Saved filenames are arch-qualified.
- Cache keys are arch-qualified.

Example output names:

- mygroup_gfx1030.amdgpu.bc
- mygroup_gfx1100.amdgpu.bc

### RDC consistency

-fgpu-rdc must be applied consistently across all three device artifact classes: embedded OSL shadeops bitcode, renderer support-library bitcode, and per-group LLVM IR emitted by BackendLLVM::run(). If any of the three is compiled without RDC while another uses it, llvm-link will fail on relocation-model mismatch.

The CMake layer should define:

```cmake
set(OSL_HIP_RDC_FLAG "-fgpu-rdc")
```

and apply it in every MAKE_HIP_BITCODE invocation.

At runtime, BackendLLVM::configure_module_target() should set the matching module state:

```cpp
if (target.backend == GPUBackendKind::AMDGPU && target.rdc) {
    ll.module()->setPICLevel(llvm::PICLevel::BigPIC);
    ll.module()->addModuleFlag(llvm::Module::Override,
                               "amdhsa_code_object_version",
                               target.code_obj_version);
}
```

For OptiX and NVPTX, rdc remains false because OptiX links at PTX level rather than at LLVM IR level.

## OSL Device Entry ABI

The device entry point ABI is stable across GPU backends. NVPTX and AMDGPU use identical parameter types and order.

```cpp
extern "C" __device__ void osl_layer_group_<GROUP>_name_<LAYER>(
    ShaderGlobals* sg,
    void* groupdata,
    void* userdata_base,
    void* output_base,
    int shade_index,
    void* interactive_params);

extern "C" __device__ void osl_init_group_<GROUP>(
    ShaderGlobals* sg,
    void* groupdata,
    void* userdata_base,
    void* output_base,
    int shade_index,
    void* interactive_params);
```

Entry point names can be queried today through group_entry_name and group_init_name. For AMDGPU, those names must not carry the __direct_callable__ prefix because that is OptiX-only. group_entry_name should therefore use api=false when the active backend is AMDGPU.

The API should also describe multiple entry layers explicitly. group_entry_name returns the last layer today, but OSL also supports explicit entry_layers and num_entry_layers. The generic artifact metadata should therefore record one exported symbol per entry layer rather than implying a single callable entry.

Device-side GroupData allocation should use both size and alignment once llvm_groupdata_alignment is added. A conservative temporary default such as 64-byte alignment is reasonable for hipMalloc, but it should not become the long-term ABI contract.

For inspection and validation, the artifact payload can be parsed back as LLVM bitcode and its exported function signatures examined directly.

## Core Source Changes

### ShadingSystemImpl

ShadingSystemImpl should own a GPUTargetDesc rather than only m_use_optix. Initial selection should move from:

```cpp
m_use_optix(renderer->supports("OptiX"))
```

to a descriptor-based choice such as:

```cpp
if (renderer->supports("NVPTX")) {
    m_gpu_target.backend = GPUBackendKind::NVPTX;
    m_gpu_target.artifact = GPUArtifactKind::PTX;
    m_gpu_target.triple = "nvptx64-nvidia-cuda";
} else if (renderer->supports("AMDGPU")) {
    m_gpu_target.backend = GPUBackendKind::AMDGPU;
    m_gpu_target.artifact = GPUArtifactKind::LLVMBitcode;
    m_gpu_target.triple = "amdgcn-amd-amdhsa";
}
```

That descriptor then flows through BackendLLVM and llvm_instance.cpp.

ShadingSystemImpl::attribute(...) should gain backend-neutral controls such as gpu_backend, gpu_artifact_format, gpu_archs, and gpu_triple, plus a backend-qualified renderer device-library injection mechanism. ShadingSystemImpl::getattribute(...) should gain gpu_num_artifacts and per-artifact queries while preserving ptx_compiled_version as a compatibility alias that reads from the new artifact container.

### ShaderGroup and renderer device libraries

ShaderGroup should store generic compiled artifacts and use a backend-neutral cache key such as m_gpu_cache_key rather than a PTX-only field. Renderer device-library injection should also move away from a single lib_bitcode slot. The API needs backend and arch in the contract so that mixed-backend or multi-arch processes are not ambiguous.

One workable shape is:

```cpp
struct RendererDeviceLibraryDesc {
    GPUBackendKind backend = GPUBackendKind::None;
    const char* arch = "";
    int size = 0;
    const void* data = nullptr;
};
```

### BackendLLVM

BackendLLVM should stop caching backend state as a boolean and should add helper predicates over shadingsys().m_gpu_target:

```cpp
bool is_gpu_backend() const
{
    return shadingsys().m_gpu_target.backend != GPUBackendKind::None;
}

bool is_nvptx_backend() const
{
    return shadingsys().m_gpu_target.backend == GPUBackendKind::NVPTX;
}

bool is_amdgpu_backend() const
{
    return shadingsys().m_gpu_target.backend == GPUBackendKind::AMDGPU;
}
```

Keeping use_optix() temporarily as a compatibility wrapper is reasonable during migration.

### llvm_instance.cpp

llvm_instance.cpp is the main implementation seam and should be reorganized around target-driven phases:

```cpp
bool BackendLLVM::prepare_seed_module(const GPUTargetDesc& target);
bool BackendLLVM::link_builtin_device_library(const GPUTargetDesc& target);
bool BackendLLVM::link_renderer_device_library(const GPUTargetDesc& target,
                                               string_view arch);
bool BackendLLVM::configure_module_target(const GPUTargetDesc& target,
                                          string_view arch);
bool BackendLLVM::emit_group_artifacts(const GPUTargetDesc& target);
```

The CPU path remains unchanged and still uses MCJIT. The NVPTX path keeps the current flow but stores PTX through the generic artifact container. The AMDGPU path loops over requested archs, reconfigures per arch, emits .bc or .ll, and appends one artifact per arch.

Per-arch module cloning is required. A single llvm::Module cannot be retargeted repeatedly because setDataLayout and setTargetTriple mutate module state. The canonical optimized module should be cloned once per arch before final artifact emission.

```cpp
bool BackendLLVM::emit_group_artifacts(const GPUTargetDesc& target)
{
    if (!group().does_nothing())
        ll.do_optimize();

    if (target.backend == GPUBackendKind::AMDGPU) {
        for (llvm::Function& fn : *ll.module())
            if (fn.hasFnAttribute("osl-lib-function"))
                fn.deleteBody();
    }

    for (const auto& arch : target.archs) {
        auto arch_module = llvm::CloneModule(*ll.module());
        arch_module->setDataLayout(target.data_layout);
#if OSL_LLVM_VERSION >= 210
        arch_module->setTargetTriple(llvm::Triple(target.triple));
#else
        arch_module->setTargetTriple(target.triple);
#endif
        if (target.rdc)
            arch_module->setPICLevel(llvm::PICLevel::BigPIC);

        GPUEmitDesc desc;
        desc.backend = target.backend;
        desc.artifact = target.artifact;
        desc.triple = target.triple;
        desc.cpu = arch;
        desc.data_layout = target.data_layout;
        desc.rdc = target.rdc;

        std::vector<uint8_t> payload;
        if (!ll.emit_gpu_artifact(desc, arch_module.get(), payload))
            return false;

        CompiledGPUArtifact art;
        art.backend = target.backend;
        art.artifact = target.artifact;
        art.triple = target.triple;
        art.arch = arch;
        art.rdc = target.rdc;
        art.llvm_version = LLVM_VERSION_STRING;
        art.exports.push_back({ GPUExportKind::Init, "",
                                init_function_name(shadingsys(), group(),
                                                   false) });
        for (int layer = 0; layer < group().nlayers(); ++layer) {
            if (!group().is_entry_layer(layer))
                continue;
            ShaderInstance* inst = group()[layer];
            art.exports.push_back({ GPUExportKind::EntryLayer,
                                    inst->layername().string(),
                                    layer_function_name(group(), *inst, false) });
        }
        art.payload = std::move(payload);
        group().m_compiled_gpu_artifacts.push_back(std::move(art));
    }
    return true;
}
```

### llvm_util.h and llvm_util.cpp

src/include/OSL/llvm_util.h should add backend-neutral enums and GPU emission option structs, keep TargetISA for CPU JIT, and expose target_machine_for(...) and emit_gpu_artifact(...). src/liboslexec/llvm_util.cpp should preserve the existing PTX emission body as a backend-specific helper while adding emit_amdgpu_bitcode(...) through WriteBitcodeToFile and an optional emit_amdgpu_ir(...) for debug output.

## Tooling Changes

### testshade

testshade is the right bring-up tool because it already acts as a compile and inspection driver. It should add emit-only AMD workflows before runtime execution. Recommended CLI additions are:

- --amdgpu or --amdgcn
- --save-amdgpu
- Repeatable --amdgpu-arch gfx1100
- --amdgpu-format bitcode|llvmir

Recommended behavior:

- Selecting --amdgpu configures the shading system target descriptor.
- --save-amdgpu writes one file per generated artifact.
- Emit-only mode succeeds even without a HIP runtime.
- --optix and --amdgpu are rejected together.

Example save logic:

```cpp
for (const auto& artifact : artifacts) {
    std::string ext = artifact.artifact == GPUArtifactKind::LLVMBitcode
        ? ".bc"
        : artifact.artifact == GPUArtifactKind::LLVMIR ? ".ll" : ".bin";
    std::string filename = fmtformat("{}_{}.{}{}", groupname, artifact.arch,
                                     backend_name(artifact.backend), ext);
    Filesystem::write_bytes(filename, artifact.payload);
}
```

### testrender and HIP

testrender should not grow a second hard-coded GPU path beside OptiX. Instead, it should be reorganized around a small GPU runtime seam.

Recommended structure:

```cpp
class GPURaytracer : public SimpleRaytracer {
public:
    virtual void create_modules() = 0;
    virtual void create_shaders() = 0;
    virtual void upload_scene() = 0;
    virtual void launch_frame() = 0;
};

class OptixRaytracer final : public GPURaytracer { ... };
class HipRaytracer final : public GPURaytracer { ... };
```

The point is to isolate GPU runtime responsibilities such as device memory management, module and program creation, shader-group artifact loading, and launch mechanics.

testrender also needs a backend-neutral shader-module abstraction because OptiX consumes PTX while HIP will want either monolithic LLVM bitcode that still needs final code-object generation or a ready-to-load HSACO.

```cpp
struct GPUShaderModuleDesc {
    GPUBackendKind backend = GPUBackendKind::None;
    GPUArtifactKind artifact = GPUArtifactKind::None;
    std::string arch;
    std::vector<GPUExportedSymbol> exports;
    std::vector<uint8_t> payload;
};

class GPURaytracer : public SimpleRaytracer {
public:
    virtual bool load_renderer_support_modules() = 0;
    virtual bool load_group_module(const GPUShaderModuleDesc& module) = 0;
    virtual bool resolve_export(const GPUExportedSymbol& symbol) = 0;
};
```

HipRaytracer::supports() should advertise explicit AMD capabilities such as GPU, AMDGPU, gpu_device_memory, and gpu_renderer_bitcode.

HIP support in testrender should also distinguish two deliverables:

1. Artifact-consumer mode.
2. Executable GPU renderer mode.

Artifact-consumer mode can stop at loading group metadata and saving or validating payloads. Executable GPU renderer mode additionally requires a final runtime artifact, likely HSACO, a launch-kernel build path, and a device-link story between renderer support code and shader-generated code.

Recommended staged testrender rollout:

1. Add a HipRaytracer shell that reports AMDGPU support and can load group artifacts without launching them.
2. Add artifact save and inspection utilities in testrender.
3. Add renderer support-library bitcode embedding for HIP.
4. Add a minimal HIP launch path for a trivial shading kernel.
5. Only then add scene-traversal parity work.

## CMake and Build-System Changes

### Top-level options

The top-level build should distinguish three concerns:

1. Enabling AMDGPU artifact emission in OSL core.
2. Enabling HIP runtime integration in sample renderers.
3. Choosing whether runtime code objects such as HSACO are built.

Recommended options:

```cmake
option(OSL_ENABLE_AMDGPU "Enable AMDGPU artifact generation in OSL core" OFF)
option(OSL_USE_HIP "Enable HIP runtime integration in sample tools" OFF)
option(OSL_BUILD_HIP_RUNTIME "Build HIP runtime code objects such as HSACO" OFF)

set(OSL_HIP_TARGET_ARCHS "gfx1100" CACHE STRING
    "Semicolon-separated AMDGPU architectures")
set(OSL_AMDGPU_ARTIFACT_FORMATS "bitcode" CACHE STRING
    "Semicolon-separated AMDGPU artifact kinds to emit: bitcode;llvmir;hsaco")
set(OSL_HSACO_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/hsaco"
    CACHE STRING "Directory where HIP code objects will be installed")
```

Expected behavior:

- OSL_ENABLE_AMDGPU=ON is enough for emit-only support in liboslexec and testshade.
- OSL_USE_HIP=ON implies OSL_ENABLE_AMDGPU=ON.
- OSL_BUILD_HIP_RUNTIME=ON requires OSL_USE_HIP=ON.

Validation example:

```cmake
if (OSL_USE_HIP AND NOT OSL_ENABLE_AMDGPU)
    set(OSL_ENABLE_AMDGPU ON CACHE BOOL "" FORCE)
endif()

if (OSL_BUILD_HIP_RUNTIME AND NOT OSL_USE_HIP)
    message(FATAL_ERROR
        "OSL_BUILD_HIP_RUNTIME requires OSL_USE_HIP=ON")
endif()
```

### Toolchain discovery

src/cmake/externalpackages.cmake should detect ROCm tooling centrally rather than ad hoc inside each helper. When AMDGPU support is enabled, configuration should verify that LLVM_TARGETS contains AMDGPU and should resolve at least:

- HIPCC_EXECUTABLE
- ROCM_CLANG_EXECUTABLE or an equivalent ROCm-aware clang++
- LLVM_LINK_TOOL
- LLVM_AS_TOOL
- LLVM_OPT_TOOL
- Optionally LLVM_LLC_TOOL
- Optionally CLANG_OFFLOAD_BUNDLER

Example validation:

```cmake
if (OSL_ENABLE_AMDGPU)
    string(FIND "${LLVM_TARGETS}" "AMDGPU" amdgpu_index)
    if (NOT amdgpu_index GREATER -1)
        message(FATAL_ERROR
            "AMDGPU target is not available in the provided LLVM build")
    endif()

    find_program(ROCM_CLANG_EXECUTABLE NAMES clang++ hipcc REQUIRED)
    find_program(LLVM_LINK_TOOL NAMES llvm-link REQUIRED)
    find_program(LLVM_AS_TOOL NAMES llvm-as REQUIRED)
    find_program(LLVM_OPT_TOOL NAMES opt REQUIRED)
endif()
```

### hip_macros.cmake

The AMD path should mirror the current two-pipeline model rather than pretending there is only one device build path. That means one path for embedded device bitcode and a separate optional path for runtime code objects.

src/cmake/hip_macros.cmake should be a new file. A reasonable first-pass surface is:

```cmake
set(OSL_HIP_EXTRA_ARGS "" CACHE STRING
    "Custom args passed to the HIP/ROCm compiler")

function(MAKE_HIP_BITCODE src suffix generated_bc extra_clang_args arch headers)
    # compile one source to AMDGPU-targeted LLVM bitcode for one arch
endfunction()

function(HIP_LINK_BITCODE output_bc input_bcs)
    # link and optionally optimize one arch-specific set of bc files
endfunction()

function(HIP_EMBED_BITCODE symbol_name output_cpp input_bc)
    # serialize bc to generated cpp using serialize-bc.py
endfunction()

function(HIP_SHADEOPS_COMPILE prefix output_bc input_srcs headers arch)
    # compile many inputs for one arch, then link to one bc
endfunction()

function(HIP_COMPILE_KERNEL hip_src extra_headers generated_hsaco extra_args arch)
    # optional later-stage runtime code object generation
endfunction()
```

The first milestone should prioritize embedded bitcode over HSACO generation.

MAKE_HIP_BITCODE(...) should compile one source file to one arch-qualified .bc output with clang++ --target=amdgcn-amd-amdhsa --offload-arch=<gfx...> and explicit header dependencies. HIP_LINK_BITCODE(...) should link all per-source .bc files for one arch and optionally run opt, using a temporary unoptimized file rather than mutating in place. HIP_SHADEOPS_COMPILE(...) should compile many inputs for one arch and link them into one per-arch library blob. HIP_COMPILE_KERNEL(...) should stay optional and gated behind OSL_BUILD_HIP_RUNTIME.

Generated filenames and embedded symbol names should be arch-qualified from the start. Example conventions:

| Stage | Filename | C++ extern name |
|---|---|---|
| Linked bitcode | shadeops_amdgpu_gfx1100.bc | intermediate only |
| Generated C++ | shadeops_amdgpu_llvm_compiled_ops_gfx1100.bc.cpp | generated file |
| Array symbol | generated in C++ | unsigned char shadeops_amdgpu_llvm_compiled_ops_gfx1100_block[] |
| Size symbol | generated in C++ | int shadeops_amdgpu_llvm_compiled_ops_gfx1100_size |
| Runtime code object | shadeops_amdgpu_gfx1100.hsaco | not embedded |

serialize-bc.py appends _block and _size automatically, so extern declarations have to match that scheme exactly.

### Per-target CMake changes

src/liboslexec/CMakeLists.txt should gain AMD build logic parallel to the current CUDA logic under USE_LLVM_BITCODE. It should compile OSL device shadeops for each requested AMD arch, serialize each linked AMD bitcode blob into generated C++, and add those generated C++ sources to lib_src. AMD names should be explicit, such as shadeops_amdgpu_<arch>.bc and shadeops_amdgpu_<arch>.bc.cpp, rather than overloading CUDA-oriented symbols.

src/testshade/CMakeLists.txt should keep host-side testshade_llvm_compiled_rs and add AMD renderer support-library bitcode generation only if testshade is expected to link renderer helpers into emitted AMD modules. HSACO should not be required for the first milestone.

src/testrender/CMakeLists.txt should add hipraytracer sources, compile HIP renderer support-library bitcode per arch, and optionally compile HSACO for later runtime milestones. Renderer support-library symbol names should be backend-qualified, such as rend_lib_nvptx_llvm_compiled_ops and rend_lib_amdgpu_llvm_compiled_ops_<arch>.

src/cmake/testing.cmake should add emit-only tests before any HIP execution tests. Good initial classes are amdgpu.emit.bc, amdgpu.emit.ll, and amdgpu.multiarch.emit. These can be implemented as testshade runs that validate artifact generation and file output without requiring a GPU runtime on CI workers.

## Caching and Artifact Lifetime

Cache keys must include backend identity and target architecture. The minimum AMD-capable key should include backend kind, artifact kind, target triple, architecture, RDC flag, LLVM version, compiler version used to build the embedded shadeops blob, optimization level used at embed time, serialized shader-group state, and renderer support-library identity or hash.

The embedded-blobs compiler version should also be baked into module metadata so stale cache entries can be detected explicitly.

Artifacts live on ShaderGroup. Their lifetime should be at least the lifetime of the optimized group. Renderer consumers must not assume textual payloads, and compatibility aliases such as ptx_compiled_version should remain clearly transitional.

## Backend Audit Checklist

Before broad AMD implementation, every current use_optix() branch should be classified into generic GPU behavior, NVPTX-specific behavior, OptiX runtime behavior, or behavior also required for AMDGPU. The main audit targets are src/liboslexec/llvm_instance.cpp, src/liboslexec/llvm_gen.cpp, src/liboslexec/backendllvm.h, and src/liboslexec/instance.cpp. This is a required implementation task rather than optional cleanup because silent divergence between OptiX and AMD semantics will otherwise accumulate.

## Implementation Order

The work should land in architecture-first order:

1. Add GPUTargetDesc and thread it through ShadingSystemImpl and BackendLLVM.
2. Replace PTX-only group storage with a generic compiled-artifact container.
3. Refactor llvm_instance.cpp into target-driven seed, link, configure, and emit phases.
4. Make LLVM_Util expose backend-neutral target-machine and artifact-emission APIs.
5. Add AMDGPU bitcode generation to the build system for OSL shadeops.
6. Add testshade emit-only AMD CLI and file-save support.
7. Add renderer support-library build flow for testshade and testrender.
8. Add cache keys that include backend and arch metadata.
9. Add testrender HIP scaffolding and artifact loading.
10. Only then decide whether in-tree HIP execution is in scope.

Once the core ABI lands, the rest is parallelizable. One developer should first land the narrow serial foundation: GPUTargetDesc, generic ShaderGroup GPU artifact storage, and the ShadingSystemImpl and BackendLLVM plumbing that replaces the boolean use_optix() model. After that, parallel tracks can cover llvm_instance.cpp refactoring, LLVM_Util and CMake/HIP pipeline work, testshade CLI and tests, and testrender HIP scaffolding. The shared merge points are the artifact query contract, backend-qualified renderer library APIs, and cache-key format, so those interfaces should be specified early and treated as fixed contracts.

## Practical First Milestone

The most realistic first deliverable is:

- Select AMDGPU as a backend target.
- Generate one bitcode artifact per requested gfx... arch.
- Store those artifacts on the ShaderGroup.
- Save them from testshade.
- Optionally expose them from testrender.
- Verify them in a downstream HIP or LLVM toolchain step.

That milestone validates the architecture while avoiding premature commitment to HIP runtime ABI details.

## Summary

Adding AMDGCN is feasible, but the hard part is not changing the target triple or teaching llc a new backend. The hard part is that OSL's current GPU path is an OptiX and NVPTX implementation threaded through core code and sample renderers. The maintainable path is to introduce a backend descriptor, make compiled artifacts generic, split generic GPU compilation from OptiX runtime rules, make AMDGPU bitcode the primary AMD artifact, and extend testshade and testrender only after that contract exists. If AMD support is added as another boolean alongside use_optix(), it may work initially, but it will be difficult to maintain, cache correctly, and extend across renderers.