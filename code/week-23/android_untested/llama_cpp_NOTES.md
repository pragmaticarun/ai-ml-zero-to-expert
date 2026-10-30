# llama.cpp on Android -- UNTESTED notes (from ggml-org/llama.cpp docs/android.md, read 2026-10-29)

- The repo ships an Android Studio project, `examples/llama.android`, with a Kotlin binding over the C/C++ library (JNI inside).
  The doc says it detects CPU features at run time and loads matching kernels; models are GGUF files in app-private storage.
- Alternative: build the CLI in Termux, or cross-compile with CMake and the NDK toolchain
  (`-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DGGML_NATIVE=OFF ...`).
- The doc warns to keep context size reasonable at first, or memory can spike: Day 158's KV arithmetic is how you pick it.
- Not tried here: no NDK, no C++ compiler on this box, no GGUF model downloaded.
