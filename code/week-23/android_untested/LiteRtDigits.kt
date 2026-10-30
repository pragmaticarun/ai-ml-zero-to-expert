// UNTESTED, and the model is missing: this box did not produce a .tflite (no PyTorch->LiteRT converter installed).
// Shape of the LiteRT CompiledModel Kotlin API as shown on ai.google.dev/edge/litert/android (imports omitted: see that page).
class LiteRtDigits(tflitePath: String) {
    private val model = CompiledModel.create(tflitePath, CompiledModel.Options(Accelerator.CPU))
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()
    fun predict(pixels: FloatArray): Int {
        inputs.get(0).writeFloat(pixels)
        model.run(inputs, outputs)
        val logits = outputs.get(0).readFloat()
        return logits.indices.maxBy { logits[it] }
    }
}
