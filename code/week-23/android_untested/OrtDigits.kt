// UNTESTED. ONNX Runtime Java API (ai.onnxruntime) used from Kotlin. Model: w23_dyn.ort copied into app storage.
// Input name "x" shape [1, 64] float32 (pixels/16), output "logits" [1, 10] -- as exported in code/week-23/_dep.py.
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class OrtDigits(modelPath: String) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    // Day 155/161 used 1 intra-op thread; the Java method for that is NOT on the pages above -- look it up in the ORT Java API reference.
    private val opts = OrtSession.SessionOptions()
    private val session = env.createSession(modelPath, opts)
    fun predict(pixels: FloatArray): Int {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(pixels), longArrayOf(1, 64)).use { t ->
            session.run(mapOf("x" to t)).use { r ->
                val logits = (r[0].value as Array<FloatArray>)[0]
                return logits.indices.maxBy { logits[it] }
            }
        }
    }
    override fun close() { session.close(); opts.close() }
}
