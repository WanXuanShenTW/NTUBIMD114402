@file:OptIn(androidx.media3.common.util.UnstableApi::class)

package com.example.myapplication

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.video.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.*
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import android.os.Build
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var recordIndicator: View
    private lateinit var headerOverlay: View
    private lateinit var backButton: ImageView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var recordVoiceButton: ImageButton

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var isOverlayVisible = false
    private var isRecording = false
    private var isVoiceRecording = false

    private var audioRecord: AudioRecord? = null
    private val REQUEST_CODE_PERMISSIONS = 1001
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private var previousEmbedding: FloatArray? = null
    private var currentSegment = mutableListOf<ByteArray>()
    private var lastSegmentTime = 0L
    private lateinit var statusText: android.widget.TextView

    @Volatile private var monitorShouldRun = false
    private var monitorThread: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d("DeviceABI", "Supported ABIs: ${Build.SUPPORTED_ABIS.joinToString()}")

        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        recordIndicator = findViewById(R.id.recordIndicator)
        headerOverlay = findViewById(R.id.headerOverlay)
        backButton = findViewById(R.id.backButton)
        recordVoiceButton = findViewById(R.id.saveVoiceButton)
        statusText = findViewById(R.id.statusText)

        recordVoiceButton.setOnClickListener {
            // 已註冊：不觸發錄音，改提示「長按可重置」
            if (hasRegisteredElder()) {
                Toast.makeText(this, "已註冊：長按可重置", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (!isVoiceRecording) {
                recordVoiceButton.setBackgroundResource(R.drawable.mic_circle_background)
                isVoiceRecording = true
                recordAndShowDialog {
                    recordVoiceButton.setBackgroundResource(android.R.color.transparent)
                    isVoiceRecording = false
                }
            }
        }

        // 長按麥克風 → 重置註冊
        recordVoiceButton.setOnLongClickListener {
            AlertDialog.Builder(this)
                .setTitle("重置長輩聲音")
                .setMessage("確定要清除已註冊的聲紋嗎？")
                .setPositiveButton("清除") { _, _ -> resetRegistration() }
                .setNegativeButton("取消", null)
                .show()
            true
        }

        previewView.setOnClickListener {
            isOverlayVisible = !isOverlayVisible
            if (isOverlayVisible) {
                headerOverlay.apply {
                    alpha = 0f
                    visibility = View.VISIBLE
                    animate().alpha(1f).setDuration(300).start()
                }
            } else {
                headerOverlay.animate().alpha(0f).setDuration(300).withEndAction {
                    headerOverlay.visibility = View.GONE
                }.start()
            }
        }

        backButton.setOnClickListener {
            val intent = Intent(this, DashboardActivity::class.java)
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            startActivity(intent)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // 一次性重置：啟動時先刪掉已註冊聲紋（跑一次後請刪掉這段）
        try {
            // 如果 ElderEmbeddingStorage 是存這個路徑與檔名（預設為這個）
            File(getExternalFilesDir(null), "elder_embedding.vec").delete()

            // 若你有用 SharedPreferences 旗標，也順便清掉（沒有也不會壞）
            getSharedPreferences("speaker", MODE_PRIVATE)
                .edit().putBoolean("elder_registered", false).apply()

            Log.d("MainActivity", "【一次性重置】已清除聲紋與註冊旗標")
        } catch (e: Exception) {
            Log.e("MainActivity", "一次性重置失敗", e)
        }

        // 註冊狀態驅動 UI 與背景監聽
        setRegisteredUI(hasRegisteredElder())
    }

    private fun updateStatus(text: String) {
        runOnUiThread {
            statusText.text = "狀態：$text"
        }
    }

    private fun isValidEmbedding(e: FloatArray?): Boolean {
        if (e == null || e.isEmpty()) return false
        var norm = 0.0
        for (v in e) {
            if (v.isNaN()) return false
            norm += (v * v)
        }
        return norm > 1e-6
    }

    private fun hasRegisteredElder(): Boolean {
        val e = ElderEmbeddingStorage.load(this)
        val ok = isValidEmbedding(e)
        val norm2 = e?.fold(0.0) { acc, v -> acc + v * v } ?: -1.0
        Log.d("RegCheck", "hasRegisteredElder? len=${e?.size ?: -1}, norm2=$norm2, ok=$ok")
        return ok
    }

    private fun setRegisteredUI(registered: Boolean) {
        if (registered) {
            recordVoiceButton.setImageResource(R.drawable.ic_mic_registered)
            recordVoiceButton.setBackgroundResource(R.drawable.mic_registered_background)
            recordVoiceButton.isEnabled = true       // ← 保持可點擊/長按
            recordVoiceButton.isLongClickable = true // ← 保證可長按
            updateStatus("已註冊，待機中")
            if (!monitorShouldRun) startBackgroundVoiceMonitor()
        } else {
            recordVoiceButton.setImageResource(R.drawable.ic_mic_white)
            recordVoiceButton.setBackgroundResource(android.R.color.transparent)
            recordVoiceButton.isEnabled = true
            updateStatus("請先註冊聲音")
            stopBackgroundVoiceMonitor()
        }
    }

    private fun startBackgroundVoiceMonitor() {
        if (monitorShouldRun) return
        monitorShouldRun = true
        monitorThread = thread(start = true, isDaemon = true) {
            val sampleRate = 16000
            val frameMs = 1000L
            val silenceNeed = 3
            val silenceThreshold = 12.0

            var silentFrames = 0
            var nonElderFrames = 0
            var segFrames = 0
            currentSegment.clear()

            fun finalizeSegment(reason: String) {
                if (currentSegment.isNotEmpty()) {
                    mergeAndSave(currentSegment)
                    updateStatus("儲存聲音段落 ✔（$reason，長度約 ${segFrames}s）")
                    currentSegment.clear()
                    segFrames = 0
                }
                silentFrames = 0
                nonElderFrames = 0
            }

            try {
                while (monitorShouldRun) {
                    val elder = ElderEmbeddingStorage.load(this@MainActivity)
                    if (!isValidEmbedding(elder)) {
                        updateStatus("尚未註冊長輩聲紋，待機中")
                        silentFrames = 0
                        nonElderFrames = 0
                        currentSegment.clear()
                        try { Thread.sleep(800) } catch (_: InterruptedException) { break }
                        continue
                    }

                    updateStatus("正在錄音...")
                    val pcmBuffer = ByteArrayOutputStream()
                    val pcm = recordPcm(frameMs, pcmBuffer)
                    val isSilent = isSilentPcm(pcm, silenceThreshold)

                    if (isSilent) {
                        silentFrames++
                        nonElderFrames = 0
                        updateStatus("偵測到靜音（${silentFrames}/${silenceNeed}）")
                    } else {
                        silentFrames = 0
                        // 建臨時檔做推論：放 cacheDir，結束一定刪掉
                        val tempFile = File(cacheDir, "chunk_${System.currentTimeMillis()}.wav")
                        try {
                            saveAsWavFile(pcm, tempFile, sampleRate, 1, 16)
                            val verifier = SpeakerVerifier(this@MainActivity)
                            updateStatus("提取聲紋中...")
                            val embedding = verifier.extractEmbedding(tempFile)

                            val isElder = if (isValidEmbedding(embedding) && isValidEmbedding(elder)) {
                                verifier.isElderVoice(embedding)
                            } else false

                            updateStatus(if (isElder) "是長輩聲音" else "不是長輩聲音")

                            if (isElder) {
                                currentSegment.add(pcm)
                                segFrames++
                                nonElderFrames = 0
                            } else {
                                // 連續 2 秒不是長輩，就把已累積的段落存起來
                                nonElderFrames++
                            }
                        } finally {
                            // 確保不留垃圾檔
                            if (tempFile.exists()) tempFile.delete()
                        }
                    }

                    // 結段條件：達靜音門檻 或 連續 2 秒不是長輩
                    if ((silentFrames >= silenceNeed || nonElderFrames >= 2) && currentSegment.isNotEmpty()) {
                        val reason = if (silentFrames >= silenceNeed) "靜音" else "非長輩"
                        finalizeSegment(reason)
                    }

                    try { Thread.sleep(200) } catch (_: InterruptedException) { break }
                }
            } catch (e: InterruptedException) {
                Log.d("VoiceMonitor", "監聽執行緒中斷（正常結束）")
            } catch (e: Exception) {
                Log.e("VoiceMonitor", "背景監聽發生錯誤", e)
                updateStatus("背景監聽錯誤：${e.message ?: e.javaClass.simpleName}")
            } finally {
                // 若停止時還有未存的片段，幫你收尾存檔
                if (currentSegment.isNotEmpty()) {
                    mergeAndSave(currentSegment)
                    Log.d("VoiceMonitor", "停止前收尾：已儲存最後一段")
                    currentSegment.clear()
                }
                if (!monitorShouldRun) {
                    updateStatus(if (hasRegisteredElder()) "已註冊，待機中" else "請先註冊聲音")
                }
            }
        }
    }

    private fun stopBackgroundVoiceMonitor() {
        monitorShouldRun = false
        monitorThread?.interrupt()
        monitorThread = null
    }

    // 一鍵重置：清除已註冊聲紋與狀態
    private fun resetRegistration() {
        try {
            // 嘗試刪外部檔
            val f1 = File(getExternalFilesDir(null), "elder_embedding.vec")
            if (f1.exists()) f1.delete()
        } catch (e: Exception) {
            Log.e("MainActivity", "刪除外部聲紋檔案失敗", e)
        }
        try {
            // 嘗試刪內部檔（若 ElderEmbeddingStorage 存在這裡）
            val f2 = File(filesDir, "elder_embedding.vec")
            if (f2.exists()) f2.delete()
        } catch (e: Exception) {
            Log.e("MainActivity", "刪除內部聲紋檔案失敗", e)
        }

        // 如後續有用 SharedPreferences 旗標，也可在這裡順便清掉
        // getSharedPreferences("speaker", MODE_PRIVATE).edit().putBoolean("elder_registered", false).apply()

        stopBackgroundVoiceMonitor()
        previousEmbedding = null
        currentSegment.clear()
        setRegisteredUI(false) // 切回未註冊 UI
        Toast.makeText(this, "已重置為未註冊狀態", Toast.LENGTH_SHORT).show()
    }

    private fun recordPcm(durationMs: Long, output: ByteArrayOutputStream): ByteArray {
        val sampleRate = 16000
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val buffer = ByteArray(bufferSize)
        recorder.startRecording()
        val start = System.currentTimeMillis()

        while (System.currentTimeMillis() - start < durationMs) {
            val read = recorder.read(buffer, 0, buffer.size)
            if (read > 0) output.write(buffer, 0, read)
        }

        recorder.stop()
        recorder.release()

        return output.toByteArray()
    }

    private fun mergeAndSave(segments: List<ByteArray>) {
        val merged = ByteArrayOutputStream()
        for (s in segments) merged.write(s)

        val file = File(getExternalFilesDir(null), "merged_${System.currentTimeMillis()}.wav")
        saveAsWavFile(merged.toByteArray(), file, 16000, 1, 16)

        Log.d("VoiceMonitor", "已合併並儲存聲音段：${file.name}")
    }

    private fun isSilentPcm(pcm: ByteArray, threshold: Double): Boolean {
        if (pcm.isEmpty()) return true
        var sum = 0.0
        var count = 0
        var i = 0
        while (i + 1 < pcm.size) {
            val low = pcm[i].toInt() and 0xFF
            val high = pcm[i + 1].toInt()
            val sample = (high shl 8) or low
            sum += kotlin.math.abs(sample.toDouble())
            count++
            i += 2
        }
        val avg = if (count > 0) sum / count else 0.0
        Log.d("VoiceMonitor", "平均振幅：$avg（門檻：$threshold）")
        return avg < threshold
    }

    private fun recordAndShowDialog(onFinish: () -> Unit) {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        val pcmData = ByteArrayOutputStream()
        val buffer = ByteArray(bufferSize)

        audioRecord?.startRecording()

        thread {
            val durationMillis = 10000L
            val startTime = System.currentTimeMillis()

            while (System.currentTimeMillis() - startTime < durationMillis) {
                val readBytes = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (readBytes > 0) {
                    pcmData.write(buffer, 0, readBytes)
                }
            }

            audioRecord?.stop()
            audioRecord?.release()

            val wavFile = File(getExternalFilesDir(null), "elder_sample.wav")
            saveAsWavFile(pcmData.toByteArray(), wavFile, sampleRate, 1, 16)

            runOnUiThread {
                onFinish()
                showConfirmDialog(wavFile)
            }
        }
    }

    private fun showConfirmDialog(wavFile: File) {
        val builder = AlertDialog.Builder(this)
            .setTitle("確認聲音樣本")
            .setMessage("要播放剛錄製的聲音嗎？")
            .setPositiveButton("播放", null)        // 先設 null，等 show 後接手點擊，不自動關閉
            .setNegativeButton("重新錄製", null)     // 同上
            .setNeutralButton("確認儲存", null)      // 同上

        val dialog = builder.create()
        dialog.setCanceledOnTouchOutside(false)

        dialog.setOnShowListener {
            var player: MediaPlayer? = null
            fun stopPlayer() {
                try { player?.stop() } catch (_: Exception) {}
                try { player?.release() } catch (_: Exception) {}
                player = null
            }

            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                val btnP = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                val btnN = dialog.getButton(AlertDialog.BUTTON_NEGATIVE)
                val btnU = dialog.getButton(AlertDialog.BUTTON_NEUTRAL)
                btnP.isEnabled = false; btnN.isEnabled = false; btnU.isEnabled = false

                thread {
                    var ok = false
                    var msg: String? = null
                    try {
                        val verifier = SpeakerVerifier(this@MainActivity)
                        val emb = verifier.extractEmbedding(wavFile)

                        fun n2(v: FloatArray) = v.fold(0.0) { a, x -> a + x * x }
                        Log.d("SaveEmbedding", "extracted len=${emb.size}, norm2=${n2(emb)}")

                        if (!isValidEmbedding(emb)) {
                            msg = "錄音太短/太小聲，未產生有效聲紋"
                        } else {
                            ElderEmbeddingStorage.save(this@MainActivity, emb)
                            val re = ElderEmbeddingStorage.load(this@MainActivity)
                            Log.d("SaveEmbedding", "reloaded len=${re?.size ?: -1}, valid=${isValidEmbedding(re)}")
                            ok = isValidEmbedding(re)
                            if (!ok) msg = "存檔後讀回無效（檔案或路徑問題）"
                        }
                    } catch (e: Exception) {
                        Log.e("SaveEmbedding", "儲存流程失敗", e)
                        msg = e.message ?: e.javaClass.simpleName
                    }

                    runOnUiThread {
                        if (ok) {
                            setRegisteredUI(true)
                            dialog.dismiss()
                            Toast.makeText(this@MainActivity, "已儲存聲紋 ✔", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(this@MainActivity, "儲存失敗：$msg", Toast.LENGTH_LONG).show()
                            btnP.isEnabled = true; btnN.isEnabled = true; btnU.isEnabled = true
                        }
                    }
                }
            }

            // 重新錄製：先關掉對話框與播放器，然後套用與首頁一致的灰底 UI，再開始錄
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                stopPlayer()
                dialog.dismiss()

                // 與點擊麥克風一致的 UI 狀態
                recordVoiceButton.setBackgroundResource(R.drawable.mic_circle_background)
                isVoiceRecording = true

                recordAndShowDialog {
                    recordVoiceButton.setBackgroundResource(android.R.color.transparent)
                    isVoiceRecording = false
                }
            }

            // 確認儲存：存 embedding -> 切到已註冊 UI -> 關閉
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                stopPlayer()
                val verifier = SpeakerVerifier(this)
                val embedding = verifier.extractEmbedding(wavFile)
                ElderEmbeddingStorage.save(this, embedding)
                setRegisteredUI(true)
                dialog.dismiss()
            }
        }

        dialog.show()
    }

    private fun saveAsWavFile(pcm: ByteArray, file: File, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val header = ByteArrayOutputStream()

        header.write("RIFF".toByteArray())
        header.write(intToByteArray(36 + pcm.size))
        header.write("WAVEfmt ".toByteArray())
        header.write(intToByteArray(16))
        header.write(shortToByteArray(1))
        header.write(shortToByteArray(channels.toShort()))
        header.write(intToByteArray(sampleRate))
        header.write(intToByteArray(byteRate))
        header.write(shortToByteArray((channels * bitsPerSample / 8).toShort()))
        header.write(shortToByteArray(bitsPerSample.toShort()))
        header.write("data".toByteArray())
        header.write(intToByteArray(pcm.size))

        FileOutputStream(file).use { fos ->
            fos.write(header.toByteArray())
            fos.write(pcm)
        }
    }

    private fun intToByteArray(value: Int) = byteArrayOf(
        (value and 0xff).toByte(),
        ((value shr 8) and 0xff).toByte(),
        ((value shr 16) and 0xff).toByte(),
        ((value shr 24) and 0xff).toByte()
    )

    private fun shortToByteArray(value: Short) = byteArrayOf(
        (value.toInt() and 0xff).toByte(),
        ((value.toInt() shr 8) and 0xff).toByte()
    )

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetRotation(previewView.display.rotation)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.SD))
                .build()

            videoCapture = VideoCapture.withOutput(recorder)

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture
                )
                startRecording()
            } catch (e: Exception) {
                Log.e("CameraX", "啟動相機失敗", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val videoFile = File(cacheDir, "video_$timestamp.mp4")
        val outputOptions = FileOutputOptions.Builder(videoFile).build()

        try {
            recording = videoCapture.output
                .prepareRecording(this, outputOptions)
                .start(ContextCompat.getMainExecutor(this)) { event ->
                    when (event) {
                        is VideoRecordEvent.Start -> {
                            isRecording = true
                            Log.d("VideoCapture", "開始錄影：${videoFile.name}")
                        }
                        is VideoRecordEvent.Finalize -> {
                            isRecording = false
                            if (event.hasError()) {
                                Log.e("VideoCapture", "錄影失敗：${event.error}")
                            } else {
                                Log.d("VideoCapture", "錄影完成：${videoFile.absolutePath}")
                            }
                            startRecording()
                        }
                    }
                }

            cameraExecutor.execute {
                Thread.sleep(5000)
                runOnUiThread {
                    recording?.stop()
                }
            }

        } catch (e: SecurityException) {
            Log.e("VideoCapture", "權限錯誤：${e.message}")
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                // 權限不足
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopBackgroundVoiceMonitor()
        cameraExecutor.shutdown()
        recording?.close()
        audioRecord?.stop()
        audioRecord?.release()
    }
}
