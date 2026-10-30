# android_untested (Week 23, Day 159) -- UNTESTED

Nothing in this folder has been compiled or run. This box has no Android SDK/NDK, no Gradle, no emulator and no phone.
Each API call below is copied in shape from the official page named next to it (checked 2026-10-29); class and method names
come from those pages only. Versions marked `<version>` are left for you to fill from the page on the day you build.

| file | runtime | model file produced on this box (Day 159, tested on x86 Linux only) | official page |
|---|---|---|---|
| build.gradle.kts.snippet | all three | - | see the per-runtime rows |
| OrtDigits.kt | ONNX Runtime Mobile (Java API from Kotlin) | out/day159_ort/w23_dyn.ort | https://onnxruntime.ai/docs/install/ , https://onnxruntime.ai/docs/get-started/with-java.html |
| EtDigits.kt | ExecuTorch Android AAR | out/w23_mlp_xnnpack.pte | https://docs.pytorch.org/executorch/stable/using-executorch-android.html |
| LiteRtDigits.kt | LiteRT CompiledModel API | (none: .tflite conversion UNTESTED) | https://ai.google.dev/edge/litert/android |
| llama_cpp_NOTES.md | llama.cpp via its Android binding / NDK | (none) | https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md |
