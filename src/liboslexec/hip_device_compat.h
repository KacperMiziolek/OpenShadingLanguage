// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

// ponytail: OIIO bit.h uses __funnelshift_lc under __HIP_DEVICE_COMPILE__; pull in
// ROCm's device intrinsics before any OIIO headers in HIP bitcode TUs.
#if defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
#    include <hip/amd_detail/amd_device_functions.h>
#endif
