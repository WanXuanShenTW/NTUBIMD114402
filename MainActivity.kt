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
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import android.os.Build
import android.speech.SpeechRecognizer
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.os.Handler
import android.os.Looper
import java.text.SimpleDateFormat
import android.content.ComponentName
import android.speech.RecognitionService
import android.media.AudioManager
import android.content.Context

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

    private fun transcriptFile(): File =
        File(getExternalFilesDir(null) ?: filesDir, "stt_transcript.jsonl")

    private fun appendTranscript(text: String, type: String = "final") {
        if (text.isBlank()) return
        val obj = org.json.JSONObject().apply {
            put("ts", System.currentTimeMillis())
            put("text", text)
            put("type", type)  // "final" 或 "partial"
        }
        val f = transcriptFile()
        android.util.Log.d("STT", "appendTranscript -> ${f.absolutePath}  type=$type  text=$text")
        java.io.FileOutputStream(f, /* append = */ true).bufferedWriter(Charsets.UTF_8).use {
            it.appendLine(obj.toString())
        }
    }

    private fun startSttLoop() {
        if (sttLoopEnabled) return
        sttLoopEnabled = true

        sttEmitAllowed = true
        sttGateDeadline = 0L
        sttTriggeredByMonitor = false
        sttOneShot = false
        sttLastPartial = ""

        stopBackgroundVoiceMonitor()
        if (stt == null) initStt()
        updateStatus("辨識中…")

        // 立刻啟動（你已改對）
        sttHandler.post { startSttOnce() }
        sttHandler.removeCallbacks(sttWatchdog)
        sttHandler.postDelayed(sttWatchdog, 4000L)
    }

    private fun startSttIfPermitted() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED) {
            startSttLoop()
        } else {
            updateStatus("等待麥克風授權…")
        }
    }

    private var previousEmbedding: FloatArray? = null
    private var currentSegment = mutableListOf<ByteArray>()
    private var lastSegmentTime = 0L
    private lateinit var statusText: android.widget.TextView
    private var sttLastPartial: String = ""

    @Volatile private var monitorShouldRun = false

    private var monitorThread: Thread? = null

    private var bgAudioRecord: AudioRecord? = null

    private var stt: SpeechRecognizer? = null
    private var isSttRunning = false
    private var sttLoopEnabled = false
    private var sttLastEventTs = 0L
    private val sttHandler = Handler(Looper.getMainLooper())

    @Volatile private var videoLoopEnabled = false      // 是否持續輪錄
    @Volatile private var isStartingRecording = false   // 正在呼叫 start 的過程
    @Volatile private var sttTriggeredByMonitor = false  // 這次 STT 是被背景偵測觸發
    @Volatile private var sttOneShot = false             // 這次只做一輪（句子結束就停）
    private val videoHandler = Handler(Looper.getMainLooper())
    private val videoStartRunnable = Runnable { tryStartRecording() }
    @Volatile private var sttReady = false
    @Volatile private var partialFlushed = false
    @Volatile private var sttEmitAllowed = false
    @Volatile private var sttGateDeadline = 0L
    private val unlockStreakNeed = 2
    @Volatile private var lastElderAt = 0L
    private val emitHoldMs = 3000L
    private val minElderFramesToUnlock = 6

    private var sttLang = "zh-TW"
    private var noMatchStreak = 0

    private var clientErrStreak = 0         // 連續 ERROR_CLIENT 次數
    private var useMinimalSttIntent = false // 是否改用極簡 Intent

    private var isGoogleStt = false

    private var lastPartialWritten: String = ""
    private var lastPartialWrittenAt: Long = 0L
    private val partialWriteMinIntervalMs = 250L

    private fun scheduleNextClip(delayMs: Long = 0L) {
        videoHandler.removeCallbacks(videoStartRunnable)
        if (videoLoopEnabled) videoHandler.postDelayed(videoStartRunnable, delayMs)
    }

    private val sttWatchdog = object : Runnable {
        override fun run() {
            if (!sttLoopEnabled) return
            val now = System.currentTimeMillis()
            // 超過 8 秒沒事件或目前沒在跑 → 重啟一次
            if (!isSttRunning || now - sttLastEventTs > 8000L) {
                try { stt?.cancel() } catch (_: Exception) {}
                startSttOnce()
            }
            sttHandler.postDelayed(this, 4000L)
        }
    }

    companion object {
        const val ACTION_STT_UPDATE = "com.example.myapplication.ACTION_STT_UPDATE"
        const val EXTRA_STT_TEXT = "EXTRA_STT_TEXT"
        const val EXTRA_STT_IS_PARTIAL = "EXTRA_STT_IS_PARTIAL"
    }

    private fun broadcastStt(text: String, partial: Boolean) {
        if (text.isBlank()) return
        val intent = Intent(ACTION_STT_UPDATE).apply {
            setPackage(packageName)                 // 限定接收者在同包
            putExtra(EXTRA_STT_TEXT, text)
            putExtra(EXTRA_STT_IS_PARTIAL, partial)
        }
        sendBroadcast(intent)
    }

    private fun touchLastElder() {
        lastElderAt = System.currentTimeMillis()
    }

    private fun startSttFromMonitor() {
        stopBackgroundVoiceMonitor()

        sttReady = false
        sttTriggeredByMonitor = true
        sttOneShot = true
        sttLoopEnabled = true

        touchLastElder()

        if (stt == null) initStt()
        updateStatus("辨識中…")

        // 立刻啟動
        sttHandler.post { startSttOnce() }
        sttHandler.removeCallbacks(sttWatchdog)
        sttHandler.postDelayed(sttWatchdog, 4000L)
    }

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

        // 註冊狀態驅動 UI 與背景監聽
        setRegisteredUI(hasRegisteredElder())
        if (allPermissionsGranted()) {
            startCamera()
            startSttIfPermitted()          // ★ 取得錄音權限後才啟 STT
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        setRegisteredUI(hasRegisteredElder())
    }

    private fun updateStatus(text: String) {
        runOnUiThread {
            statusText.text = "狀態：$text"
        }
    }

    private fun isDevicePlaying(): Boolean {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return am.isMusicActive
    }

    private fun initStt() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Toast.makeText(this, "此裝置不支援語音辨識", Toast.LENGTH_SHORT).show()
            return
        }

        val svc = pickRecognizer()

        stt = try {
            if (svc != null) SpeechRecognizer.createSpeechRecognizer(this, svc)
            else SpeechRecognizer.createSpeechRecognizer(this)
        } catch (_: Exception) {
            SpeechRecognizer.createSpeechRecognizer(this)
        }

        isGoogleStt = svc?.packageName == "com.google.android.googlequicksearchbox" ||
                svc?.packageName == "com.google.android.apps.gsa"

        val pkg = svc?.packageName ?: "DEFAULT"
        Log.d("STT", "Using recognizer: $pkg")

        stt?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                clientErrStreak = 0
                sttLastEventTs = System.currentTimeMillis()
                sttReady = true

                updateStatus("請開始說話")

                touchLastElder()

                if (sttTriggeredByMonitor) {
                    val deadline = sttGateDeadline.takeIf { it > 0 } ?: (System.currentTimeMillis() + 2800L)
                    sttHandler.post(object : Runnable {
                        override fun run() {
                            if (!sttLoopEnabled) return
                            when {
                                sttEmitAllowed -> {
                                    stopBackgroundVoiceMonitor()
                                }
                                System.currentTimeMillis() >= deadline -> {
                                    try { stt?.cancel() } catch (_: Exception) {}
                                    stopSttLoop()
                                    return
                                }
                                else -> sttHandler.postDelayed(this, 120L)
                            }
                        }
                    })
                }
                sttEmitAllowed = true
                sttGateDeadline = 0L
                lastElderAt = System.currentTimeMillis()
            }

            override fun onBeginningOfSpeech() {
                sttLastEventTs = System.currentTimeMillis()
                noMatchStreak = 0
                touchLastElder()
            }

            override fun onRmsChanged(rmsdB: Float) {
                sttLastEventTs = System.currentTimeMillis()
                if (isSttRunning) touchLastElder()
            }

            override fun onBufferReceived(buffer: ByteArray?) { sttLastEventTs = System.currentTimeMillis() }

            override fun onEndOfSpeech() {
                isSttRunning = false
                sttLastEventTs = System.currentTimeMillis()
                if (sttTriggeredByMonitor && sttOneShot) {
                    stopSttLoop()
                } else if (sttLoopEnabled) {
                    sttHandler.postDelayed({ startSttOnce() }, 250)
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                sttLastEventTs = System.currentTimeMillis()

                val text = partialResults
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()
                if (text.isBlank()) return

                sttLastPartial = text

                val now = System.currentTimeMillis()
                if (text != lastPartialWritten || now - lastPartialWrittenAt >= partialWriteMinIntervalMs) {
                    broadcastStt(text, true)
                    lastPartialWritten = text
                    lastPartialWrittenAt = now
                }
                noMatchStreak = 0
            }

            override fun onResults(results: Bundle?) {
                sttLastEventTs = System.currentTimeMillis()

                val text = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()

                if (text.isNotBlank()) {
                    sttLastPartial = ""
                    broadcastStt(text, false)
                    appendTranscript(text, "final")
                    N8nSender.sendElderVoice(this@MainActivity, text)
                }

                isSttRunning = false
                if (sttTriggeredByMonitor && sttOneShot) {
                    stopSttLoop()
                } else if (sttLoopEnabled) {
                    sttHandler.postDelayed({ startSttOnce() }, 250)
                }
            }

            override fun onError(error: Int) {
                isSttRunning = false
                sttLastEventTs = System.currentTimeMillis()

                if (sttTriggeredByMonitor) {
                    sttEmitAllowed = false
                    sttGateDeadline = 0L
                }

                if (sttTriggeredByMonitor && sttOneShot &&
                    (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT)) {
                    stopSttLoop()
                    return
                }

                Log.e("STT", "onError=$error (${sttErrorName(error)})")

                when (error) {
                    // 客戶端錯誤：維持你原本的 reset 流程
                    SpeechRecognizer.ERROR_CLIENT -> {
                        clientErrStreak++
                        sttEmitAllowed = false
                        sttGateDeadline = 0L

                        if (clientErrStreak == 1) {
                            useMinimalSttIntent = false
                            safeResetStt(800)
                        } else if (clientErrStreak == 2) {
                            useMinimalSttIntent = true
                            safeResetStt(800)
                        } else {
                            try { stt?.destroy() } catch (_: Exception) {}
                            stt = null
                            initStt()
                            clientErrStreak = 0
                            if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 1000)
                        }
                    }

                    // 這兩種最常見：聽不到或太短 → 簡單延遲再啟，不再切語言
                    SpeechRecognizer.ERROR_NO_MATCH,
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                        noMatchStreak++
                        if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 600)
                    }

                    // 麥克風暫時被占：冷卻一下再試
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                        if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 700)
                    }

                    // 伺服器類：照你原本邏輯 reset
                    SpeechRecognizer.ERROR_SERVER -> {
                        sttEmitAllowed = false
                        sttGateDeadline = 0L
                        safeResetStt(1000)
                    }

                    else -> {
                        if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 1000)
                    }
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                sttLastEventTs = System.currentTimeMillis()
            }
        })
    }

    private fun forceReleaseMic() {
        try { audioRecord?.stop() } catch (_: Exception) {}
        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null

        try { bgAudioRecord?.stop() } catch (_: Exception) {}
        try { bgAudioRecord?.release() } catch (_: Exception) {}
        bgAudioRecord = null

        monitorShouldRun = false
    }

    private fun pickRecognizer(): ComponentName? {
        val pm = packageManager
        val query = pm.queryIntentServices(
            Intent(RecognitionService.SERVICE_INTERFACE),
            PackageManager.MATCH_ALL
        )
        val realStt = query.filter { it.serviceInfo.permission == "android.permission.BIND_SPEECH_RECOGNITION_SERVICE" }
        val google = realStt.firstOrNull {
            it.serviceInfo.packageName == "com.google.android.googlequicksearchbox" ||
                    it.serviceInfo.packageName == "com.google.android.apps.gsa"
        }
        val chosen = google ?: realStt.firstOrNull()
        return chosen?.let { ComponentName(it.serviceInfo.packageName, it.serviceInfo.name) }
    }

    private fun buildSttIntent(): Intent =
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)

            // 先固定一個語言讓流程穩定；通了再考慮動態切換
            sttLang = "zh-Hant-TW"           // 或 "cmn-Hant-TW" 二擇一
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, sttLang)

            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)

            // 一些裝置需要 calling package 才會穩
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
            }

            if (isGoogleStt) {
                putExtra("android.speech.extra.DICTATION_MODE", true)
            }

            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, false)

            // ⚠️ 先不要設定靜音時間，先用系統預設（更容易成功）
            // putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2000)
            // putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1500)
            // putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 1200)
        }

    override fun onResume() {
        super.onResume()
        videoLoopEnabled = false
        if (allPermissionsGranted()) {
            if (videoCapture == null) startCamera()
        } else {
            updateStatus("等待授權中…")
        }
    }

    override fun onPause() {
        super.onPause()

        sttEmitAllowed = false
        sttGateDeadline = 0L

        videoLoopEnabled = false
        videoHandler.removeCallbacksAndMessages(null)

        try { recording?.stop() } catch (_: Exception) {}
        try { recording?.close() } catch (_: Exception) {}
        recording = null
        isRecording = false
        isStartingRecording = false

        try { ProcessCameraProvider.getInstance(this).get().unbindAll() } catch (_: Exception) {}
        videoCapture = null

        stopBackgroundVoiceMonitor()

        if (sttLoopEnabled) stopSttLoop()
    }

    private fun stopSttLoop() {
        sttLoopEnabled = false
        isSttRunning = false
        sttOneShot = false
        sttTriggeredByMonitor = false

        sttEmitAllowed = false
        sttGateDeadline = 0L

        if (sttLastPartial.isNotBlank() && !partialFlushed) {
            partialFlushed = true
            sttLastPartial = ""
        }
        sttHandler.removeCallbacksAndMessages(null)
        try { stt?.stopListening() } catch (_: Exception) {}
        try { stt?.cancel() } catch (_: Exception) {}
        // 這會在 ~600ms 後依註冊狀態恢復背景監聽（不會搶麥）
        sttHandler.postDelayed({ setRegisteredUI(hasRegisteredElder()) }, 600L)
    }

    private fun sttErrorName(code: Int) = when (code) {
        1 -> "NETWORK_TIMEOUT"
        2 -> "NETWORK"
        3 -> "AUDIO"
        4 -> "SERVER"
        5 -> "CLIENT"
        6 -> "SPEECH_TIMEOUT"
        7 -> "NO_MATCH"
        8 -> "RECOGNIZER_BUSY"
        9 -> "INSUFFICIENT_PERMISSIONS"
        else -> "UNKNOWN($code)"
    }

    private fun safeResetStt(delay: Long) {
        sttEmitAllowed = false
        sttGateDeadline = 0L

        try { stt?.cancel() } catch (_: Exception) {}
        try { stt?.destroy() } catch (_: Exception) {}
        stt = null
        sttHandler.postDelayed({
            initStt()
            if (sttLoopEnabled) startSttOnce()
        }, delay)
    }

    private fun startSttOnce() {
        if (!sttLoopEnabled || isSttRunning) return
        partialFlushed = false
        val recognizer = stt ?: return
        try {
            forceReleaseMic()
            isSttRunning = true
            sttLastEventTs = System.currentTimeMillis()
            Log.d("STT", "startListening(lang=$sttLang)")
            recognizer.startListening(buildSttIntent())
        } catch (e: Exception) {
            isSttRunning = false
            updateStatus("啟動 STT 失敗：${e.message}")
            if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 1000)
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
            recordVoiceButton.isEnabled = true
            recordVoiceButton.isLongClickable = true
            if (sttLoopEnabled) {
                updateStatus("辨識中…")   //STT 開著 → 顯示辨識中
            } else {
                updateStatus("已註冊，待機中")
                if (!monitorShouldRun) startBackgroundVoiceMonitor()
            }
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
            val frameMs = 160L
            val silenceNeedFrames = 3
            val hangoverFrames = 2
            val nonElderEndFrames = 8
            val silentDbThreshold = -42f
            val maxSegFrames = 120
            val preFrames = 15                 // ≈2.4s pre-roll
            val startStreakNeed = 1            // 開段門檻
            val unlockStreakNeed = 2           // 解鎖需連續長輩幀數（可調 2~3）
            val minElderFramesToUnlock = 4     // ✅ 解鎖需累積長輩幀數（4 幀 ≈ 640ms）

            var silentFrames = 0
            var nonElderFrames = 0
            var nonSpeechFrames = 0
            var segFrames = 0
            var elderStreak = 0
            var elderFramesInSeg = 0
            var tailFrames = 0
            var sttRequested = false

            val pre = kotlin.collections.ArrayDeque<ByteArray>(preFrames)
            currentSegment.clear()

            fun isLikelySpeech(pcm: ByteArray, db: Float? = null): Boolean {
                val d = db ?: calcRmsDb(pcm)
                val z = zeroCrossRate(pcm)
                return d > silentDbThreshold && z in 0.02f..0.25f
            }

            fun eligibleToSave(): Boolean {
                if (currentSegment.isEmpty()) return false
                val audible = isSegmentAudible(currentSegment, silentDbThreshold + 3f)
                val ratio = if (segFrames > 0) elderFramesInSeg.toFloat() / segFrames else 0f
                return audible && (elderFramesInSeg >= 4 || ratio >= 0.30f)
            }

            fun finalizeSegment(reason: String) {
                if (eligibleToSave()) {
                    mergeAndSave(currentSegment)
                    updateStatus("儲存聲音段落（$reason，約 ${segFrames * frameMs} ms）")
                } else {
                    Log.d("VoiceMonitor", "未保存：整段無長輩語音或比例不足（$reason）")
                }
                currentSegment.clear(); pre.clear()
                silentFrames = 0; nonElderFrames = 0; nonSpeechFrames = 0
                segFrames = 0; elderStreak = 0; elderFramesInSeg = 0; tailFrames = 0
                sttRequested = false

                // 只有在沒有 STT 正在跑時才關掉輸出閘門，避免擋到當輪 STT
                if (!sttLoopEnabled || !isSttRunning) {
                    sttEmitAllowed = false
                    sttGateDeadline = 0L
                }
            }

            try {
                while (monitorShouldRun) {
                    val elder = ElderEmbeddingStorage.load(this@MainActivity)
                    if (!isValidEmbedding(elder)) {
                        updateStatus("尚未註冊長輩聲紋，待機中")
                        finalizeSegment("未註冊")
                        try { Thread.sleep(800) } catch (_: InterruptedException) { break }
                        continue
                    }

                    // 讀一幀
                    val pcm = recordPcmFromMonitor(frameMs, sampleRate)
                    if (pcm.isEmpty()) {
                        // 麥克風可能被占用，先稍等避免狂迴圈
                        try { Thread.sleep(200) } catch (_: InterruptedException) { break }
                        continue
                    }

                    val db = calcRmsDb(pcm)
                    val speech = isLikelySpeech(pcm, db)
                    val isSilent = db < silentDbThreshold

                    // 維持 pre-roll（固定長度）
                    if (pre.size == preFrames) pre.removeFirst()
                    pre.addLast(pcm)

                    if (isSilent) {
                        silentFrames++
                        nonElderFrames = 0
                        nonSpeechFrames++
                        elderStreak = 0
                        updateStatus("偵測到靜音（${silentFrames}/${silenceNeedFrames}）")
                    } else {
                        silentFrames = 0
                        nonSpeechFrames = if (speech) 0 else nonSpeechFrames + 1

                        // 只有像語音的才跑聲紋
                        val isElder = if (speech) {
                            val tmp = File(cacheDir, "chunk_${System.currentTimeMillis()}.wav")
                            try {
                                saveAsWavFile(pcm, tmp, sampleRate, 1, 16)
                                val v = SpeakerVerifier(this@MainActivity)
                                val emb = v.extractEmbedding(tmp)
                                isValidEmbedding(emb) && v.isElderVoice(emb)
                            } finally {
                                try { tmp.delete() } catch (_: Exception) {}
                            }
                        } else false

                        if (isElder) {
                            updateStatus("是長輩聲音")
                            elderStreak++
                            nonElderFrames = 0
                            tailFrames = 0

                            // 最近一次長輩時間
                            lastElderAt = System.currentTimeMillis()

                            if (segFrames == 0) {
                                // —— 首段：達到開段門檻就把 pre-roll + 當前幀納入
                                if (elderStreak >= startStreakNeed) {
                                    for (pr in pre) currentSegment.add(pr)
                                    currentSegment.add(pcm)
                                    segFrames++
                                    elderFramesInSeg++
                                }
                            } else {
                                // 段中
                                currentSegment.add(pcm)
                                segFrames++
                                elderFramesInSeg++
                            }

                            // ✅ 更嚴格解鎖：連續幀 + 累積幀 + 無外放
                            if (!sttEmitAllowed &&
                                elderStreak >= unlockStreakNeed &&
                                elderFramesInSeg >= minElderFramesToUnlock &&
                                !isDevicePlaying()
                            ) {
                                val justUnlocked = !sttEmitAllowed
                                sttEmitAllowed = true
                                Log.d("VoiceMonitor", "🔓 解鎖輸出（elderStreak=$elderStreak, elderFramesInSeg=$elderFramesInSeg）")

                                // 立刻回補目前 partial，避免句首看起來消失
                                if (justUnlocked && sttLastPartial.isNotBlank()) {
                                    broadcastStt(sttLastPartial, true)
                                    appendTranscript(sttLastPartial, "partial")
                                }

                                // ✅ 現在才啟 STT：先停監聽（釋放麥克風），再啟 STT
                                if (!sttRequested) {
                                    sttRequested = true
                                    runOnUiThread {
                                        stopBackgroundVoiceMonitor()
                                        sttHandler.postDelayed({ startSttFromMonitor() }, 150)
                                    }
                                }
                            } else if (!sttEmitAllowed &&
                                elderStreak >= unlockStreakNeed &&
                                elderFramesInSeg >= minElderFramesToUnlock &&
                                isDevicePlaying()
                            ) {
                                Log.d("VoiceMonitor", "⏸ 裝置正在播放，暫不解鎖")
                            }
                        } else {
                            updateStatus("不是長輩聲音")
                            elderStreak = 0
                            nonElderFrames++

                            if (segFrames > 0) {
                                // 段已開始：只補少量尾巴，避免長噪音進段
                                if (tailFrames < hangoverFrames) {
                                    currentSegment.add(pcm)
                                    segFrames++
                                    tailFrames++
                                }
                            }

                            // ✅ 連續非長輩幀達到門檻 → 主動關閘，避免外部語音接手
                            if (sttEmitAllowed && nonElderFrames >= 4) {
                                sttEmitAllowed = false
                                Log.d("VoiceMonitor", "🔒 關閉輸出（nonElderFrames=$nonElderFrames）")
                            }
                        }
                    }

                    val endBySilence  = silentFrames >= (silenceNeedFrames + hangoverFrames)
                    val endByNonElder = nonElderFrames >= nonElderEndFrames && nonSpeechFrames >= 4
                    val endByMax      = segFrames >= maxSegFrames

                    if ((endBySilence || endByNonElder || endByMax) && segFrames > 0) {
                        val reason = when {
                            endByMax     -> "達最大長度"
                            endBySilence -> "靜音"
                            else         -> "非長輩/非語音"
                        }
                        finalizeSegment(reason)
                    }
                }
            } catch (_: InterruptedException) {
                Log.d("VoiceMonitor", "監聽執行緒中斷（正常結束）")
            } catch (e: Exception) {
                Log.e("VoiceMonitor", "背景監聽發生錯誤", e)
                updateStatus("背景監聽錯誤：${e.message ?: e.javaClass.simpleName}")
            } finally {
                if (currentSegment.isNotEmpty()) finalizeSegment("停止前收尾")
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
        try { bgAudioRecord?.stop() } catch (_: Exception) {}
        try { bgAudioRecord?.release() } catch (_: Exception) {}
        bgAudioRecord = null
    }

    // 一鍵重置：清除已註冊聲紋與狀態
    private fun resetRegistration() {
        try {
            val f1 = File(getExternalFilesDir(null), "elder_embedding.vec")
            if (f1.exists()) f1.delete()
        } catch (e: Exception) {
            Log.e("MainActivity", "刪除外部聲紋檔案失敗", e)
        }
        try {
            val f2 = File(filesDir, "elder_embedding.vec")
            if (f2.exists()) f2.delete()
        } catch (e: Exception) {
            Log.e("MainActivity", "刪除內部聲紋檔案失敗", e)
        }

        stopBackgroundVoiceMonitor()
        previousEmbedding = null
        currentSegment.clear()
        setRegisteredUI(false)
        Toast.makeText(this, "已重置為未註冊狀態", Toast.LENGTH_SHORT).show()
    }

    private fun zeroCrossRate(pcm: ByteArray): Float {
        var crossings = 0
        if (pcm.size < 4) return 0f
        fun sampleAt(j: Int): Int {
            val lo = pcm[j].toInt() and 0xFF
            val hi = pcm[j + 1].toInt()
            return (hi shl 8) or lo
        }
        var prev = sampleAt(0)
        var i = 2
        var count = 1
        while (i + 1 < pcm.size) {
            val cur = sampleAt(i)
            if ((prev >= 0 && cur < 0) || (prev < 0 && cur >= 0)) crossings++
            prev = cur
            i += 2
            count++
        }
        return crossings.toFloat() / maxOf(1, count - 1)
    }

    private fun calcRmsDb(pcm: ByteArray): Float {
        var sumSq = 0.0
        var count = 0
        var i = 0
        while (i + 1 < pcm.size) {
            val lo = pcm[i].toInt() and 0xFF
            val hi = pcm[i + 1].toInt()
            val s = (hi shl 8) or lo
            val f = s / 32768f
            sumSq += (f * f)
            count++
            i += 2
        }
        val rms = kotlin.math.sqrt((sumSq / maxOf(1, count))).toFloat()
        val eps = 1e-8f
        val ln10 = 2.302585092994046f
        return 20f * (kotlin.math.ln(rms + eps) / ln10)
    }

    private fun isSegmentAudible(segments: List<ByteArray>, minRmsDb: Float = -42f): Boolean {
        val total = segments.sumOf { it.size }
        if (total <= 2) return false
        val out = ByteArrayOutputStream(total)
        for (s in segments) out.write(s)
        val db = calcRmsDb(out.toByteArray())
        Log.d("VoiceMonitor", "segment RMS dB=$db (min=$minRmsDb)")
        return db > minRmsDb
    }

    private fun mergeAndSave(segments: List<ByteArray>) {
        val merged = ByteArrayOutputStream()
        for (s in segments) merged.write(s)

        val dir = getExternalFilesDir(null) ?: filesDir
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, "merged_${System.currentTimeMillis()}.wav")

        val sr = 16000
        var bytes = merged.toByteArray()

        if (bytes.size < 4) {
            Log.d("VoiceMonitor", "未保存：段落為空")
            return
        }

        bytes = dcBlock(bytes, alpha = 0.995f)
        bytes = trimHeadNoise(pcm = bytes, sampleRate = sr, frameMs = 250, minDb = -41f)
        if (bytes.size < 4) {
            Log.d("VoiceMonitor", "未保存：剪噪後為空/過短")
            return
        }
        bytes = cutAtNearestZeroCrossingHead(bytes, sampleRate = sr, searchMs = 15)
        bytes = applyFadeInOut(bytes, sampleRate = sr, fadeMs = 35)

        saveAsWavFile(bytes, file, sr, 1, 16)
        Log.d("VoiceMonitor", "已合併並儲存聲音段：${file.absolutePath}")
    }

    private fun dcBlock(pcm: ByteArray, alpha: Float = 0.995f): ByteArray {
        if (pcm.size < 4) return pcm
        val out = pcm.copyOf()
        var xPrev = 0
        var yPrev = 0f
        var i = 0
        while (i + 1 < out.size) {
            val lo = out[i].toInt() and 0xFF
            val hi = out[i + 1].toInt()
            val x = (hi shl 8) or lo
            val y = (x - xPrev) + alpha * yPrev
            val s = y.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[i] = (s and 0xFF).toByte()
            out[i + 1] = ((s shr 8) and 0xFF).toByte()
            xPrev = x
            yPrev = y
            i += 2
        }
        return out
    }

    private fun trimHeadNoise(
        pcm: ByteArray,
        sampleRate: Int,
        frameMs: Int = 250,
        minDb: Float = -39f
    ): ByteArray {
        val bytesPerFrame = (sampleRate * frameMs / 1000) * 2
        if (pcm.size < bytesPerFrame) return pcm
        val n = pcm.size / bytesPerFrame
        var startIdx = 0
        for (i in 0 until n) {
            val from = i * bytesPerFrame
            val to = from + bytesPerFrame
            val frame = pcm.copyOfRange(from, to)
            val db = calcRmsDb(frame)
            val speech = db > minDb && zeroCrossRate(frame) in 0.02f..0.25f
            if (speech) { startIdx = (i - 2).coerceAtLeast(0); break }
        }
        val cutFrom = (startIdx * bytesPerFrame).coerceAtMost(pcm.size)
        return pcm.copyOfRange(cutFrom, pcm.size)
    }

    private fun applyFadeInOut(pcm: ByteArray, sampleRate: Int, fadeMs: Int = 35): ByteArray {
        if (pcm.size < 4) return pcm
        val out = pcm.copyOf()
        val totalSamples = out.size / 2
        val fadeSamples = ((sampleRate * fadeMs) / 1000).coerceAtMost(totalSamples / 2)

        for (i in 0 until fadeSamples) {
            val idx = i * 2
            val lo = out[idx].toInt() and 0xFF
            val hi = out[idx + 1].toInt()
            val s = (hi shl 8) or lo
            val w = 0.5f - 0.5f * kotlin.math.cos(Math.PI.toFloat() * (i + 1) / fadeSamples)
            val ns = (s * w).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[idx] = (ns and 0xFF).toByte()
            out[idx + 1] = ((ns shr 8) and 0xFF).toByte()
        }
        for (i in 0 until fadeSamples) {
            val idx = (totalSamples - 1 - i) * 2
            val lo = out[idx].toInt() and 0xFF
            val hi = out[idx + 1].toInt()
            val s = (hi shl 8) or lo
            val w = 0.5f - 0.5f * kotlin.math.cos(Math.PI.toFloat() * (i + 1) / fadeSamples)
            val ns = (s * w).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[idx] = (ns and 0xFF).toByte()
            out[idx + 1] = ((ns shr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun cutAtNearestZeroCrossingHead(pcm: ByteArray, sampleRate: Int, searchMs: Int = 15): ByteArray {
        if (pcm.size < 4) return pcm
        val searchSamples = ((sampleRate * searchMs) / 1000).coerceAtMost(pcm.size / 2 - 2)
        var prev = ((pcm[1].toInt() shl 8) or (pcm[0].toInt() and 0xFF))
        var cut = 0
        for (i in 1..searchSamples) {
            val idx = i * 2
            val cur = ((pcm[idx + 1].toInt() shl 8) or (pcm[idx].toInt() and 0xFF))
            if ((prev >= 0 && cur < 0) || (prev < 0 && cur >= 0)) {
                cut = idx
                break
            }
            prev = cur
        }
        return if (cut > 0) pcm.copyOfRange(cut, pcm.size) else pcm
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
            .setPositiveButton("播放", null)
            .setNegativeButton("重新錄製", null)
            .setNeutralButton("確認儲存", null)

        val dialog = builder.create()
        dialog.setCanceledOnTouchOutside(false)

        dialog.setOnShowListener {
            var player: MediaPlayer? = null

            fun stopPlayer() {
                try { player?.stop() } catch (_: Exception) {}
                try { player?.release() } catch (_: Exception) {}
                player = null
            }

            dialog.setOnDismissListener { stopPlayer() }

            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    stopPlayer()
                    player = MediaPlayer().apply {
                        setDataSource(wavFile.absolutePath)
                        prepare()
                        start()
                        setOnCompletionListener {
                            Toast.makeText(
                                this@MainActivity,
                                "播放結束，可按「確認儲存」或「重新錄製」",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, "播放失敗：${e.message}", Toast.LENGTH_SHORT).show()
                }
            }

            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                stopPlayer()
                dialog.dismiss()

                recordVoiceButton.setBackgroundResource(R.drawable.mic_circle_background)
                isVoiceRecording = true

                recordAndShowDialog {
                    recordVoiceButton.setBackgroundResource(android.R.color.transparent)
                    isVoiceRecording = false
                }
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
                            Toast.makeText(this@MainActivity, "已儲存聲紋", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(this@MainActivity, "儲存失敗：$msg", Toast.LENGTH_LONG).show()
                            btnP.isEnabled = true; btnN.isEnabled = true; btnU.isEnabled = true
                        }
                    }
                }
            }
        }

        dialog.show()
    }

    private fun saveAsWavFile(
        pcm: ByteArray,
        file: File,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {
        file.parentFile?.mkdirs()

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
                if (videoLoopEnabled) scheduleNextClip(200)
            } catch (e: Exception) {
                Log.e("CameraX", "啟動相機失敗", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun tryStartRecording() {
        val vc = this.videoCapture ?: return
        if (!videoLoopEnabled) return
        if (isStartingRecording || isRecording) return

        isStartingRecording = true

        val timestamp = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault()).format(java.util.Date())
        val videoFile = File(cacheDir, "video_$timestamp.mp4")
        val outputOptions = FileOutputOptions.Builder(videoFile).build()

        try {
            recording = vc.output
                .prepareRecording(this, outputOptions)
                .start(ContextCompat.getMainExecutor(this)) { event ->
                    when (event) {
                        is VideoRecordEvent.Start -> {
                            isStartingRecording = false
                            isRecording = true
                            Log.d("VideoCapture", "開始錄影：${videoFile.name}")

                            videoHandler.postDelayed({
                                try { recording?.stop() } catch (_: Exception) {}
                            }, 5000)
                        }
                        is VideoRecordEvent.Finalize -> {
                            isRecording = false
                            recording = null
                            val hadError = event.hasError()
                            if (hadError) {
                                Log.e("VideoCapture", "錄影失敗：${event.error}")
                            } else {
                                Log.d("VideoCapture", "錄影完成：${videoFile.absolutePath}")
                            }
                            if (videoLoopEnabled) scheduleNextClip(if (hadError) 500 else 120)
                        }
                    }
                }
        } catch (e: Exception) {
            isStartingRecording = false
            isRecording = false
            Log.e("VideoCapture", "startRecording 失敗：${e.message}")
            scheduleNextClip(600)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
                startSttIfPermitted()     // ★ 拿到授權後再啟 STT
            } else {
                updateStatus("未授權麥克風/相機")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        sttEmitAllowed = false
        sttGateDeadline = 0L

        stopSttLoop()
        try { stt?.destroy() } catch (_: Exception) {}
        stopBackgroundVoiceMonitor()
        cameraExecutor.shutdown()
        try { recording?.close() } catch (_: Exception) {}

        try { audioRecord?.stop() } catch (_: Exception) {}
        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null
    }

    private fun ensureMonitorRecorder(sampleRate: Int): AudioRecord? {
        var r = bgAudioRecord
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (r == null || r.state != AudioRecord.STATE_INITIALIZED) {
            val bufSize = maxOf(minBuf, sampleRate / 5 * 2) // ≈200ms buffer
            val tmp = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufSize
            )

            if (tmp.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("VoiceMonitor", "AudioRecord init failed (state=${tmp.state}). Mic busy?")
                try { tmp.release() } catch (_: Exception) {}
                return null
            }

            try {
                tmp.startRecording()
            } catch (e: IllegalStateException) {
                Log.e("VoiceMonitor", "startRecording() failed: ${e.message}")
                try { tmp.release() } catch (_: Exception) {}
                return null
            }

            bgAudioRecord = tmp
            r = tmp
            Log.d("VoiceMonitor", "bgAudioRecord started. minBuf=$minBuf, useBuf=$bufSize")
        } else if (r.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            try {
                r.startRecording()
            } catch (e: IllegalStateException) {
                Log.e("VoiceMonitor", "startRecording() on existing recorder failed: ${e.message}")
                try { r.release() } catch (_: Exception) {}
                bgAudioRecord = null
                return null
            }
        }
        return r
    }

    private fun recordPcmFromMonitor(durationMs: Long, sampleRate: Int): ByteArray {
        val r = ensureMonitorRecorder(sampleRate) ?: return ByteArray(0)
        val bytesToRead = ((sampleRate * durationMs) / 1000L * 2L).toInt()
        val out = ByteArray(bytesToRead)
        var off = 0
        while (off < bytesToRead && monitorShouldRun) {
            val n = r.read(out, off, bytesToRead - off)
            if (n > 0) off += n
            else if (n == 0) Thread.yield()
            else { Log.e("VoiceMonitor", "AudioRecord read error: $n"); break }
        }
        return if (off == bytesToRead) out else out.copyOf(off)
    }
}
