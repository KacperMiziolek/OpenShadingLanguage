# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Compile one C++ source to AMDGPU LLVM bitcode (.bc)
function (MAKE_HIP_BITCODE src suffix generated_bc extra_clang_args)
    get_filename_component (src_we ${src} NAME_WE)
    set (bc_hip "${CMAKE_CURRENT_BINARY_DIR}/${src_we}${suffix}.bc")
    set (${generated_bc} ${bc_hip} PARENT_SCOPE)

    get_property (CURRENT_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
    message (VERBOSE "Current #defines are ${CURRENT_DEFINITIONS}")
    foreach (def ${CURRENT_DEFINITIONS})
        set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} "-D${def}")
    endforeach ()
    set (LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS} ${CSTD_FLAGS})

    list (TRANSFORM IMATH_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_IMATH_INCLUDES)
    list (TRANSFORM OpenImageIO_INCLUDES PREPEND -I
          OUTPUT_VARIABLE ALL_OpenImageIO_INCLUDES)

    add_custom_command (OUTPUT ${bc_hip}
        COMMAND ${LLVM_BC_GENERATOR}
            "-I${CMAKE_CURRENT_SOURCE_DIR}"
            "-I${CMAKE_SOURCE_DIR}/src/liboslexec"
            "-I${CMAKE_BINARY_DIR}/include"
            "-I${PROJECT_SOURCE_DIR}/src/include"
            ${ALL_OpenImageIO_INCLUDES}
            ${ALL_IMATH_INCLUDES}
            ${LLVM_COMPILE_FLAGS} ${HIP_LIB_FLAGS}
            -include "${CMAKE_SOURCE_DIR}/src/liboslexec/hip_device_compat.h"
            -DOSL_COMPILING_TO_BITCODE=1 -DNDEBUG -DOIIO_NO_SSE
            -x hip --cuda-device-only --offload-arch=${OSL_HIP_TARGET_ARCH}
            --std=c++${CMAKE_CXX_STANDARD}
            -fno-math-errno -ffast-math -O3
            -Wno-deprecated-register -Wno-format-security
            -emit-llvm -c ${extra_clang_args}
            ${src} -o ${bc_hip}
        DEPENDS ${src} ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction ()

# Compile many sources for the AMD arch and link into one .bc
function (HIP_SHADEOPS_COMPILE prefix output_bc input_srcs headers)
    set (linked_bc "${CMAKE_CURRENT_BINARY_DIR}/linked_${prefix}.bc")
    set (${output_bc} ${linked_bc} PARENT_SCOPE)

    foreach (shadeops_src ${input_srcs})
        MAKE_HIP_BITCODE (${shadeops_src} "_amdgpu" shadeops_bc "")
        list (APPEND shadeops_bc_list ${shadeops_bc})
    endforeach ()

    add_custom_command (OUTPUT ${linked_bc}
        COMMAND ${LLVM_LINK_TOOL} ${shadeops_bc_list} -o ${linked_bc}
        COMMAND ${LLVM_OPT_TOOL} -passes="default<O3>" ${linked_bc} -o ${linked_bc}
        DEPENDS ${shadeops_bc_list} ${input_srcs} ${headers}
                ${exec_headers} ${PROJECT_PUBLIC_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction ()
