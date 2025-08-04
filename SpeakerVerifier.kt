package com.example.myapplication

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioDispatcherFactory
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor

class SpeakerVerifier(private val context: Context) {

    private lateinit var ortSession: OrtSession
    private lateinit var ortEnv: OrtEnvironment

    companion object {
        private const val TAG = "SpeakerVerifier"
    }

    init {
        try {
            initOrtModel()
            Log.d(TAG, "ONNX 模型初始化成功")
        } catch (e: Exception) {
            Log.e(TAG, "ONNX 模型初始化失敗", e)
        }
    }

    private fun initOrtModel() {
        ortEnv = OrtEnvironment.getEnvironment()
        val modelBytes = loadModelFromAssets("ecapa_tdnn.onnx")
        val sessionOptions = OrtSession.SessionOptions()
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)
        // Debug: 印出模型的輸入型態與形狀，協助確認期望維度
        try {
            val inputName = ortSession.inputNames.iterator().next()
            val info = ortSession.inputInfo[inputName] as NodeInfo
            val tInfo = info.info as TensorInfo
            Log.d(TAG, "模型輸入: name=$inputName, type=${tInfo.type}, shape=${tInfo.shape.contentToString()}")
        } catch (_: Exception) {}
    }

    private fun loadModelFromAssets(fileName: String): ByteArray {
        return context.assets.open(fileName).use { it.readBytes() }
    }

    fun extractEmbedding(audioFile: File): FloatArray {
        Log.d(TAG, "開始提取聲紋，檔案：${audioFile.absolutePath}")
        return try {
            val inputTensor = preprocessAudio(audioFile)
            if (inputTensor.isEmpty()) {
                Log.d(TAG, "偵測為靜音或無有效特徵，略過推論")
                FloatArray(0)
            } else {
                Log.d(TAG, "前處理完成，輸入長度: ${inputTensor.size}")
                val inputName = ortSession.inputNames.iterator().next()
                val shape = longArrayOf(1L, 1L, inputTensor.size.toLong()) // 預期 80
                val floatBuffer = FloatBuffer.allocate(inputTensor.size).apply {
                    put(inputTensor)
                    rewind()
                }
                val tensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
                Log.d(TAG, "Tensor 建立完成，開始推論...")
                ortSession.run(mapOf(inputName to tensor)).use { results ->
                    val output = results[0].value as Array<FloatArray>
                    Log.d(TAG, "推論完成，嵌入向量長度: ${output[0].size}")
                    val emb = output[0]
                    // 清掉 NaN/Inf，避免存檔後被判為無效
                    for (i in emb.indices) {
                        val v = emb[i]
                        if (!v.isFinite()) emb[i] = 0f
                    }
                    emb
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "提取聲紋失敗", e)
            throw RuntimeException("推論失敗：${e.message}", e)
        }
    }

    // —— 防呆：合法性檢查 ——
    private fun isValidEmbedding(e: FloatArray?): Boolean {
        if (e == null || e.isEmpty()) return false
        var norm = 0.0
        for (v in e) {
            if (v.isNaN()) return false
            norm += (v * v)
        }
        return norm > 1e-6
    }

    fun isElderVoice(currentEmbedding: FloatArray?, threshold: Float = 0.85f): Boolean {
        val savedEmbedding = ElderEmbeddingStorage.load(context)
        // 未註冊或向量無效，一律回 false
        if (!isValidEmbedding(savedEmbedding) || !isValidEmbedding(currentEmbedding)) {
            Log.d(TAG, "比對略過：無有效長輩聲紋或目前向量無效")
            return false
        }
        // 長度不一致，直接 false
        if (savedEmbedding!!.size != currentEmbedding!!.size) {
            Log.d(TAG, "比對略過：維度不一致 saved=${savedEmbedding.size}, cur=${currentEmbedding.size}")
            return false
        }

        val similarity = cosineSimilarity(savedEmbedding, currentEmbedding)
        if (similarity.isNaN() || similarity.isInfinite()) {
            Log.d(TAG, "比對略過：相似度為非數或無窮")
            return false
        }
        Log.d(TAG, "比對註冊長輩聲音，相似度：$similarity，門檻：$threshold")
        return similarity >= threshold
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            val av = a[i]
            val bv = b[i]
            dot += av * bv
            normA += av * av
            normB += bv * bv
        }
        if (normA <= 1e-6f || normB <= 1e-6f) return -1f // 避免除以零，視為極不相似
        return (dot / (sqrt(normA) * sqrt(normB))).coerceIn(-1f, 1f)
    }

    private fun preprocessAudio(audioFile: File): FloatArray {
        // 讀取 16kHz/mono/16-bit PCM WAV，資料從 44 bytes 後開始
        val bytes = audioFile.readBytes()
        if (bytes.size <= 44) return FloatArray(0)
        val dataOffset = 44
        val sampleCount = (bytes.size - dataOffset) / 2
        if (sampleCount <= 0) return FloatArray(0)

        // 錄音段靜音判斷：RMS + 峰值 + 有聲比例（三條件同時成立才視為靜音）
        var sumSq = 0.0
        var peak = 0f
        var activeCount = 0
        var j = dataOffset
        for (i in 0 until sampleCount) {
            val lo = bytes[j].toInt() and 0xFF
            val hi = bytes[j + 1].toInt()
            val s  = (hi shl 8) or lo
            val f  = (s / 32768f)
            val a  = kotlin.math.abs(f)
            if (a > peak) peak = a
            if (a > 0.02f) activeCount++ // 超過 0.02 視為「有聲」
            sumSq += (f * f)
            j += 2
        }
        val rms = kotlin.math.sqrt((sumSq / sampleCount)).toFloat()
        val rmsDb = (20f * kotlin.math.log10(rms + 1e-8f))
        val activeRatio = activeCount.toFloat() / sampleCount
        if (rmsDb < -50f && peak < 0.02f && activeRatio < 0.01f) {
            Log.d(TAG, "preprocess: 判定靜音 rmsDb=$rmsDb peak=$peak activeRatio=$activeRatio")
            return FloatArray(0)
        }

        // 取 80 維 melBands，時間平均 + 安全 log 壓縮
        val dispatcher: AudioDispatcher = AudioDispatcherFactory.fromPipe(
            audioFile.absolutePath, 16000, 1024, 512
        )
        val mfcc = be.tarsos.dsp.mfcc.MFCC(
            bufferSize = 1024,
            sampleRate = 16000,
            numberOfMelFilters = 80,
            lowerFilterFreq = 20f,
            upperFilterFreq = 8000f
        )
        val sum = FloatArray(80)
        var frames = 0
        dispatcher.addAudioProcessor(mfcc)
        dispatcher.addAudioProcessor(object : AudioProcessor {
            override fun process(audioEvent: AudioEvent): Boolean {
                val mel = mfcc.melBands
                if (mel != null && mel.size >= 80) {
                    for (i in 0 until 80) {
                        val m = mel[i]
                        val safe = if (m.isFinite() && m > 0f) m else 0f // 非有限/負值歸零
                        sum[i] += safe
                    }
                    frames++
                }
                return true
            }
            override fun processingFinished() {}
        })
        dispatcher.run()
        if (frames == 0) return FloatArray(0)
        val feat = FloatArray(80)
        for (i in 0 until 80) {
            val avg = sum[i] / frames
            val nonNeg = if (avg.isFinite() && avg > 0f) avg else 0f
            feat[i] = Math.log1p(nonNeg.toDouble()).toFloat()
            if (!feat[i].isFinite()) feat[i] = 0f
        }
        return feat
    }
}
