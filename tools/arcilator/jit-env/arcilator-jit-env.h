// Arcilator JIT runtime environment API facing the arcilator.cpp
#pragma once

#ifdef _MSC_VER
#ifdef ARCJITENV_EXPORTS
#define ARCJITENV_API extern "C" __declspec(dllexport)
#else
#define ARCJITENV_API extern "C" __declspec(dllimport)
#endif // ARCJITENV_EXPORTS
#else
#define ARCJITENV_API extern "C"
#endif // _MSC_VER

// These don't do anything at the moment. It is still
// required to call them to make sure the library
// is linked and loaded before the JIT engine starts.

ARCJITENV_API int arc_jit_runtime_env_init(void);
ARCJITENV_API int arc_jit_runtime_env_deinit(void);
