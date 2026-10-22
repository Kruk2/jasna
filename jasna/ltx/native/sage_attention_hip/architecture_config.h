// comfy-kitchen 0.2.38's architecture_config.h as its CMake generates it from architectures.json.
#pragma once

#if defined(__gfx1200__) || defined(__gfx1201__)
#define COMFY_MMA_GFX12 1
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define COMFY_MMA_GFX11 1
#endif

#if defined(COMFY_MMA_GFX12) || defined(COMFY_MMA_GFX11)
#define COMFY_HAS_WMMA 1
#endif
