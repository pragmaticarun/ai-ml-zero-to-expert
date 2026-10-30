// UNTESTED. ExecuTorch Android Java API (org.pytorch.executorch) from Kotlin. Model: w23_mlp_xnnpack.pte (static batch 1).
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class EtDigits(ptePath: String) {
    private val module = Module.load(ptePath)
    fun predict(pixels: FloatArray): Int {
        val input = EValue.from(Tensor.fromBlob(pixels, longArrayOf(1, 64)))
        val logits = module.forward(input)[0].toTensor().getDataAsFloatArray()
        return logits.indices.maxBy { logits[it] }
    }
}
