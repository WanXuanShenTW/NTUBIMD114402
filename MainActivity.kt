package com.example.myapplication

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min
import org.json.JSONArray
import org.json.JSONObject
import android.view.Surface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.speech.SpeechRecognizer
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.os.Handler
import android.os.Looper
import android.content.Intent
import android.content.ComponentName
import android.speech.RecognitionService
import android.media.AudioManager
import java.io.*
import androidx.appcompat.app.AlertDialog
import android.content.Context
import android.Manifest
import kotlin.concurrent.thread
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.graphics.Paint
import android.view.Gravity
import androidx.core.view.updateLayoutParams
import android.widget.FrameLayout
import kotlin.math.roundToInt
import android.content.IntentFilter
import android.util.Base64


class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: View
    private lateinit var btnStartInfer: Button
    private lateinit var btnStopInfer: Button
    private lateinit var btnBackToImage: Button
    private lateinit var seekFps: SeekBar
    private lateinit var tvFpsValue: TextView
    private lateinit var switchRenderBoxes: SwitchCompat
    private lateinit var switchObjectDetect: SwitchCompat
    private lateinit var switchPoseDetect: SwitchCompat
    private lateinit var classes: List<String>

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private lateinit var cameraExecutor: ExecutorService

    // 在類別欄位加兩個快取（用於 overlay）：
    private var lastPoseForOverlay: PoseResult? = null
    private var lastObjForOverlay: ObjectResult? = null
    private var poseBurstCounter = 0

    // FPS 控制
    private var targetFps: Int = 10
    private var allowProcess = false
    private var lastProcessTimeMs = 0L

    // ==== 新增：推論與傳送的時間戳與結果快取 ====
    // 推論節奏（毫秒）
    private val POSE_INFER_MS = 200L
    private val DETECT_INFER_MS = 1000L

    private var lastPoseInferAt = 0L
    private var lastDetectInferAt = 0L

    // 最後一次成功推論的結果（送包與畫面都用這份）
    @Volatile private var lastPoseResult: PoseResult? = null
    @Volatile private var lastObjResult: ObjectResult? = null

    // 送包節拍器
    private var senderHandler: Handler? = null
    private var senderRunnable: Runnable? = null
    private var senderThread: android.os.HandlerThread? = null

    @Volatile private var lastFrameW: Int = 0
    @Volatile private var lastFrameH: Int = 0


    companion object {
        private const val CAMERA_PERMISSION_CODE = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)
        private const val TAG = "CameraDetect"
        private const val TAG_STT_FINAL = "STT_FINAL"
        private const val TAG_AI_REPLY = "AI_REPLY"

        const val ACTION_STT_UPDATE = "com.example.myapplication.ACTION_STT_UPDATE"
        const val EXTRA_STT_TEXT = "EXTRA_STT_TEXT"
        const val EXTRA_STT_IS_PARTIAL = "EXTRA_STT_IS_PARTIAL"

        const val MIC_PERMISSION_CODE = 2002

        const val ACTION_AI_REPLY = "com.example.myapplication.ACTION_AI_REPLY"
        const val ACTION_USER_UTTER = "com.example.myapplication.ACTION_USER_UTTER"
        const val EXTRA_AI_TEXT = "EXTRA_AI_TEXT"
        const val EXTRA_AI_AUDIO_URL = "EXTRA_AI_AUDIO_URL"
        const val EXTRA_SESSION_ID = "EXTRA_SESSION_ID"
        const val EXTRA_ELDER_ID = "EXTRA_ELDER_ID"
    }

    // --- ONNX 相關 ---
    private lateinit var ortEnv: ai.onnxruntime.OrtEnvironment
    private lateinit var objSession: ai.onnxruntime.OrtSession
    private lateinit var poseSession: ai.onnxruntime.OrtSession
    private val objDetector = ObjectDetector()
    private val poseDetector = PoseDetector()

    private lateinit var ws: WsManager
    private var frameId: Long = 0
    // 把你的 WS 位址換成實際值（支援 ws:// 或 wss://）
    private val WS_URL = "wss://6e8e59bcc446.ngrok-free.app/ws/pose?user_id=3"

    private lateinit var switchSendWs: SwitchCompat
    @Volatile private var sendWsEnabled = false
    // 是否在 overlay 繪製（用來測吞吐）
    @Volatile private var renderEnabled = true

    // WS 節流參數
    private var lastWsSendAt = 0L
    private var wsTargetFps = 8
    // 傳輸頻率（Pose 5fps、Detect 2.5fps）
    private var lastPoseSendAt = 0L
    private var lastDetectSendAt = 0L
    private val POSE_SEND_FPS = 5.0
    private val DETECT_SEND_FPS = 2.5


    // 開關：執行哪種偵測
    @Volatile private var objectDetectEnabled = true
    @Volatile private var poseDetectEnabled = false

    // === 語音 / STT ===
    private lateinit var headerOverlay: View
    private lateinit var recordVoiceButton: ImageButton
    private lateinit var statusText: TextView

    private var stt: SpeechRecognizer? = null
    private var pausedByPlayback = false
    private var isSttRunning = false
    private var sttLoopEnabled = false
    private var sttLastEventTs = 0L
    private val sttHandler = Handler(Looper.getMainLooper())

    private lateinit var voiceStatus: TextView
    private lateinit var inferenceStatus: TextView

    @Volatile private var sttReady = false
    @Volatile private var partialFlushed = false
    @Volatile private var sttEmitAllowed = false
    @Volatile private var sttGateDeadline = 0L
    @Volatile private var sttTriggeredByMonitor = false
    @Volatile private var sttOneShot = false

    private var sttLang = "zh-Hant-TW"
    private var noMatchStreak = 0
    private var clientErrStreak = 0
    private var useMinimalSttIntent = false
    private var isGoogleStt = false

    private var lastPartialWritten: String = ""
    private var lastPartialWrittenAt: Long = 0L
    private val partialWriteMinIntervalMs = 250L
    private var sttLastPartial: String = ""

    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()

    // === 背景長輩聲音監聽 ===
    @Volatile private var monitorShouldRun = false
    private var monitorThread: Thread? = null
    private var audioRecord: AudioRecord? = null
    private var bgAudioRecord: AudioRecord? = null
    private var isVoiceRecording = false
    private var previousEmbedding: FloatArray? = null
    private var currentSegment = mutableListOf<ByteArray>()
    private var lastSegmentTime = 0L
    @Volatile private var lastElderAt = 0L
    private val unlockStreakNeed = 2
    private val minElderFramesToUnlock = 6
    private val emitHoldMs = 3000L
    @Volatile var aiSpeaking = false
    private val uiHandler = Handler(Looper.getMainLooper())
    private var aiReceiverRegistered = false

    private val audioManager by lazy { getSystemService(Context.AUDIO_SERVICE) as AudioManager }
    private var playbackFocusRequest: android.media.AudioFocusRequest? = null

    private var ttsPlayer: MediaPlayer? = null



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 綁定 View
        previewView        = findViewById(R.id.previewView)
        overlayView        = findViewById(R.id.overlayView)
        btnStartInfer      = findViewById(R.id.btnStartInfer)
        btnStopInfer       = findViewById(R.id.btnStopInfer)
        btnBackToImage     = findViewById(R.id.btnBackToImage)
        seekFps            = findViewById(R.id.seekFps)
        tvFpsValue         = findViewById(R.id.tvFpsValue)
        switchRenderBoxes  = findViewById(R.id.switchRenderBoxes)
        switchObjectDetect = findViewById(R.id.switchObjectDetect)
        switchPoseDetect   = findViewById(R.id.switchPoseDetect)
        switchSendWs       = findViewById(R.id.switchSendWs)

        classes = readClasses()
        recordVoiceButton = findViewById(R.id.saveVoiceButton)
        statusText = findViewById(R.id.statusText)
        inferenceStatus = findViewById(R.id.inferenceStatus)
        inferenceStatus.bringToFront()


        // 先初始化 Camera 執行緒，避免 startCamera() 尚未就緒
        cameraExecutor = Executors.newSingleThreadExecutor()

        // ONNX Sessions（放這裡即可）
        ortEnv = ai.onnxruntime.OrtEnvironment.getEnvironment()
        objSession  = ortEnv.createSession(readRawModel(R.raw.yolov8n), ai.onnxruntime.OrtSession.SessionOptions())
        poseSession = ortEnv.createSession(readRawModel(R.raw.yolov8n_pose), ai.onnxruntime.OrtSession.SessionOptions())

        // 語音按鈕
        recordVoiceButton.setOnClickListener {
            if (hasRegisteredElder()) {
                Toast.makeText(this, "已註冊：長按可重置", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (!isVoiceRecording) {
                isVoiceRecording = true
                setRegisteringUI(true)
                recordAndShowDialog {
                    isVoiceRecording = false
                    updateStatus("錄音完成，請確認聲音樣本")
                    setRegisteringUI(false)
                }
            }
        }

        setRegisteredUI(hasRegisteredElder())

        // 權限：相機（單獨處理）
        if (hasCameraPermission()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        }

        // 權限：麥克風（單獨處理）
        if (hasAudioPermission()) {
            initStt()
            startSttIfPermitted()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                MIC_PERMISSION_CODE
            )
        }

        // FPS SeekBar（1~30，預設10）
        seekFps.max = 30
        seekFps.progress = targetFps
        tvFpsValue.text = "$targetFps fps"
        seekFps.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val p = progress.coerceIn(1, 30)
                targetFps = p
                tvFpsValue.text = "$p fps"
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        // 顯示框（整體渲染）開關
        renderEnabled = switchRenderBoxes.isChecked
        switchRenderBoxes.setOnCheckedChangeListener { _, isChecked ->
            renderEnabled = isChecked
            if (!isChecked) overlayView.overlay.clear()
        }

        // 模式開關：物件 / 骨架
        objectDetectEnabled = switchObjectDetect.isChecked
        poseDetectEnabled   = switchPoseDetect.isChecked
        switchObjectDetect.setOnCheckedChangeListener { _, isChecked -> objectDetectEnabled = isChecked }
        switchPoseDetect.setOnCheckedChangeListener   { _, isChecked -> poseDetectEnabled   = isChecked }

        btnBackToImage.setOnClickListener { finish() }

        btnStartInfer.setOnClickListener {
            try {
                lastPoseSendAt = 0L
                lastDetectSendAt = 0L
                allowProcess = true
                lastProcessTimeMs = 0L
                if (!sendWsEnabled) {
                    // 用 post 避免同一次點擊引發 listener 裡又改 UI 造成奇怪 re-entrancy
                    switchSendWs.post { switchSendWs.isChecked = true }
                }
                Toast.makeText(this, "開始以 $targetFps fps 擷取幀", Toast.LENGTH_SHORT).show()
            } catch (t: Throwable) {
                Log.e(TAG, "start infer click crashed", t)
            }
        }

        btnStopInfer.setOnClickListener {
            allowProcess = false
            lastProcessTimeMs = 0L
            stopSenderLoop()
            if (sendWsEnabled) switchSendWs.isChecked = false
            Toast.makeText(this, "已停止", Toast.LENGTH_SHORT).show()
        }

        Thread.setDefaultUncaughtExceptionHandler { _, e ->
            Log.e(TAG, "FATAL", e)
        }

        ws = WsManager(WS_URL)

        switchSendWs.setOnCheckedChangeListener { _, isChecked ->
            sendWsEnabled = isChecked
            if (isChecked) {
                lastPoseSendAt = 0L
                lastDetectSendAt = SystemClock.elapsedRealtime()
                ws.connect(
                    onState = { ok, err -> if (!ok) Log.w(TAG, "WS connect failed: $err") },
                    onMessage = { msg -> handleServerMessage(msg) }
                )
                startSenderLoop()
            } else {
                stopSenderLoop()
                ws.close()
            }
            Thread.setDefaultUncaughtExceptionHandler { _, e ->
                Log.e(TAG, "FATAL (default handler)", e)
            }
        }
    }

    fun onAiSpeakingStart() {
        aiSpeaking = true
        try { stt?.stopListening() } catch (_: Exception) {}
        isSttRunning = false
    }

    fun onAiSpeakingDone() {
        aiSpeaking = false
        // 延遲一點點，避免剛結束還有尾音被收進來
        uiHandler.postDelayed({ startSttIfPermitted() }, 600)
    }

    // ===== 權限 =====
    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, CAMERA_PERMISSION_CODE)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            CAMERA_PERMISSION_CODE -> {
                val camOk = hasCameraPermission()
                val micOk = hasAudioPermission()

                if (camOk) {
                    startCamera()
                } else {
                    Toast.makeText(this, "未授權相機，無法開啟預覽", Toast.LENGTH_SHORT).show()
                    finish()
                    return
                }

                // 有麥克風就開 STT；沒有就繼續顯示提示（不結束 app）
                if (micOk) {
                    startSttIfPermitted()
                } else {
                    Toast.makeText(this, "未授權麥克風，語音辨識將無法使用", Toast.LENGTH_SHORT).show()
                }
            }

            MIC_PERMISSION_CODE -> {
                if (hasAudioPermission()) {
                    startSttIfPermitted()
                    Toast.makeText(this, "已授權麥克風", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "未授權麥克風，無法錄音/辨識", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    // ===== CameraX: Preview + Analysis =====
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val selector = CameraSelector.DEFAULT_BACK_CAMERA

                imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetRotation(safeRotation())
                    .build().apply {
                        setAnalyzer(cameraExecutor) { imageProxy -> analyzeFrame(imageProxy) }
                    }

                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(this, selector, preview, imageAnalysis)

            } catch (e: Exception) {
                Log.e(TAG, "startCamera error", e)
                Toast.makeText(this, "相機初始化失敗：${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    fun decodeBase64Audio(b64: String): ByteArray {
        // 去空白/換行，避免出錯
        val clean = b64.trim().replace("\\s".toRegex(), "")
        // 若來源是 URL-safe 的（含 -/_），可改用 Base64.URL_SAFE
        return Base64.decode(clean, Base64.DEFAULT)
    }

    // 1) 直接播 MP3 bytes（先存到 cache，再用 FileDescriptor 播放）
    fun playMp3Bytes(
        bytes: ByteArray,
        onDone: (() -> Unit)? = null,
        onError: ((Exception) -> Unit)? = null
    ) {
        try { ttsPlayer?.release() } catch (_: Exception) {}
        ttsPlayer = null

        onAiSpeakingStart()
        pauseVoiceStuffForPlayback()

        try {
            val f = File(cacheDir, "tts_${System.currentTimeMillis()}.mp3")
            var fis: FileInputStream? = null

            // 確保檔案完整寫入
            FileOutputStream(f).use { fos ->
                fos.write(bytes)
                fos.flush()
                fos.fd.sync()
            }

            val p = MediaPlayer()
            ttsPlayer = p

            // 一定要在 prepared/complete 後才關閉 fis
            fis = FileInputStream(f)
            p.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            p.setDataSource(fis.fd)

            p.setOnPreparedListener { mp ->
                // (可選) audio focus
                // requestPlaybackFocus()
                mp.start()
            }
            p.setOnCompletionListener { mp ->
                try { fis?.close() } catch (_: Exception) {}
                try { f.delete() } catch (_: Exception) {}
                // abandonPlaybackFocus()
                mp.release()
                ttsPlayer = null
                resumeVoiceStuffAfterPlayback()
                onAiSpeakingDone()
                onDone?.invoke()
            }
            p.setOnErrorListener { mp, what, extra ->
                try { fis?.close() } catch (_: Exception) {}
                try { f.delete() } catch (_: Exception) {}
                // abandonPlaybackFocus()
                mp.release()
                ttsPlayer = null
                resumeVoiceStuffAfterPlayback()
                onAiSpeakingDone()
                onError?.invoke(RuntimeException("MediaPlayer error $what/$extra"))
                true
            }
            p.prepareAsync()

        } catch (e: Exception) {
            resumeVoiceStuffAfterPlayback()
            onAiSpeakingDone()
            onError?.invoke(e)
        }
    }

    // MainActivity 內
    fun handleTtsResponse(bodyString: String) {
        val trimmed = bodyString.trim()

        val rootObj = if (trimmed.startsWith("[")) {
            val arr = org.json.JSONArray(trimmed)
            if (arr.length() == 0) throw IllegalArgumentException("empty array")
            arr.getJSONObject(0)
        } else {
            org.json.JSONObject(trimmed)
        }

        val audioObj = rootObj.getJSONObject("audio_data")
        val b64 = audioObj.getString("data")

        val clean = b64.trim().replace("\\s".toRegex(), "")
        val mp3Bytes = android.util.Base64.decode(clean, android.util.Base64.DEFAULT)

        playMp3Bytes(
            bytes = mp3Bytes,
            onError = { e -> Log.e("TTS", "play bytes failed", e) }
        )
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val now = SystemClock.elapsedRealtime()
            val interval = (1000L / targetFps.coerceAtLeast(1))
            val shouldProcess = allowProcess && (now - lastProcessTimeMs >= interval)
            if (shouldProcess) {
                lastProcessTimeMs = now
                val bmp = imageProxyToBitmapRGBA(image)
                try {
                    processFrame(bmp)  // ← 包起來
                } catch (t: Throwable) {
                    Log.e(TAG, "processFrame crashed", t)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyzeFrame error", e)
        } finally {
            image.close()
        }
    }

    private fun imageProxyToBitmapRGBA(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val srcW = image.width
        val srcH = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride // 對 RGBA_8888 會是 4

        val buffer = plane.buffer
        buffer.rewind()

        // 把每一列的有效像素（srcW * pixelStride）拷到連續的大陣列
        val rowBytes = srcW * pixelStride
        val all = ByteArray(srcH * rowBytes)
        val rowTmp = ByteArray(rowStride)

        var dstOff = 0
        repeat(srcH) {
            buffer.get(rowTmp, 0, rowStride)
            // 只取前面有效的部分（忽略 padding）
            System.arraycopy(rowTmp, 0, all, dstOff, rowBytes)
            dstOff += rowBytes
        }

        val bmp = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(ByteBuffer.wrap(all))

        // 依裝置回報角度旋正
        val degrees = image.imageInfo.rotationDegrees
        if (degrees == 0) return bmp
        val m = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        if (rotated != bmp) bmp.recycle()
        return rotated
    }

    // ===== 檔案與廣播 =====
    private fun transcriptFile(): File {
        val sp = getSharedPreferences("app", Context.MODE_PRIVATE)
        val elderId = sp.getInt("elder_id", -1)
        val fname = if (elderId > 0) {
            "stt_transcript_elder${elderId}.jsonl"
        } else {
            "stt_transcript.jsonl"
        }
        return File(getExternalFilesDir(null) ?: filesDir, fname)
    }

    // MainActivity.kt
    private fun appendTranscript(text: String, type: String = "final") {
        if (text.isBlank() && type != "ai") return

        val obj = JSONObject().apply {
            put("ts", System.currentTimeMillis())
            put("text", text)
            put("type", type)
        }
        val f = transcriptFile()
        FileOutputStream(f, true).bufferedWriter(Charsets.UTF_8).use {
            it.appendLine(obj.toString())
        }
    }


    private fun broadcastStt(text: String, partial: Boolean) {
        if (text.isBlank()) return
        val intent = Intent(ACTION_STT_UPDATE).apply {
            setPackage(packageName)
            putExtra(EXTRA_STT_TEXT, text)
            putExtra(EXTRA_STT_IS_PARTIAL, partial)
        }
        sendBroadcast(intent)
    }

    // ===== UI & 狀態 =====
    private fun updateStatus(text: String) {
        runOnUiThread { statusText.text = "狀態：$text" }
    }

    private fun touchLastElder() { lastElderAt = System.currentTimeMillis() }

    private fun isDevicePlaying(): Boolean {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return am.isMusicActive
    }

    // ===== STT 初始化與 Intent =====
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
            sttLang = "zh-Hant-TW"
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, sttLang)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
            }
            if (isGoogleStt) {
                putExtra("android.speech.extra.DICTATION_MODE", true)
            }
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, false)
            // 靜音時間維持系統預設（更容易成功）
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
                showMicListeningIcon()
                touchLastElder()

                if (sttTriggeredByMonitor) {
                    val deadline = sttGateDeadline.takeIf { it > 0 } ?: (System.currentTimeMillis() + 5000L)
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
                if (aiSpeaking || pausedByPlayback) return

                sttLastEventTs = System.currentTimeMillis()
                val text = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
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
                if (aiSpeaking || pausedByPlayback) return

                // ✅ 門沒開就不處理結果、也不送到 N8n
                if (!sttEmitAllowed) return

                sttLastEventTs = System.currentTimeMillis()
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                if (text.isNotBlank()) {
                    sttLastPartial = ""
                    broadcastStt(text, false)
                    appendTranscript(text, "final")
                    val sessionId = SessionId.next(this@MainActivity)
                    val elderId = getSharedPreferences("app", Context.MODE_PRIVATE).getInt("elder_id", 1)

                    Log.i(TAG_STT_FINAL, "sid=$sessionId elder=$elderId text=$text")

                    val userMsg = Intent(ACTION_USER_UTTER).apply {
                        setPackage(packageName)
                        putExtra(EXTRA_ELDER_ID, elderId)
                        putExtra(EXTRA_SESSION_ID, sessionId)
                        putExtra(EXTRA_AI_TEXT, text)
                    }
                    sendBroadcast(userMsg)

                    N8nSender.sendElderVoiceAndSpeak(this@MainActivity, text, sessionId)
                }

                isSttRunning = false
                if (sttTriggeredByMonitor && sttOneShot) stopSttLoop()
                else if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 250)
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
                    SpeechRecognizer.ERROR_NO_MATCH,
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                        noMatchStreak++
                        if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 600)
                    }
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                        if (sttLoopEnabled) sttHandler.postDelayed({ startSttOnce() }, 700)
                    }
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

    fun showAiAnswer(text: String) {
        runOnUiThread { statusText.text = "AI：$text" }
    }

    // 2) 播放 URL 或本地路徑：本地路徑用 FileDescriptor，並保持 fis 開著
    fun playTtsFromUrl(url: String, answerText: String) {
        runOnUiThread {
            try { ttsPlayer?.release() } catch (_: Exception) {}
            ttsPlayer = null

            onAiSpeakingStart()
            pauseVoiceStuffForPlayback()

            val p = MediaPlayer()
            ttsPlayer = p

            var fis: FileInputStream? = null
            p.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )

            if (url.startsWith("/")) {
                // 本地檔案：用 FileDescriptor，比字串路徑穩
                val f = File(url)
                fis = FileInputStream(f)
                p.setDataSource(fis.fd)
            } else {
                // 遠端 URL 照舊
                p.setDataSource(url)
            }

            p.setOnPreparedListener { mp ->
                // requestPlaybackFocus()
                mp.start()
            }
            p.setOnCompletionListener { mp ->
                try { fis?.close() } catch (_: Exception) {}
                // abandonPlaybackFocus()
                mp.release()
                ttsPlayer = null
                resumeVoiceStuffAfterPlayback()
                onAiSpeakingDone()
            }
            p.setOnErrorListener { mp, what, extra ->
                try { fis?.close() } catch (_: Exception) {}
                // abandonPlaybackFocus()
                Log.e("TTS", "播放錯誤: $what/$extra")
                mp.release()
                ttsPlayer = null
                resumeVoiceStuffAfterPlayback()
                onAiSpeakingDone()
                true
            }
            p.prepareAsync()
        }
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

    private val aiReplyReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != ACTION_AI_REPLY) return

            val text = intent.getStringExtra(EXTRA_AI_TEXT).orEmpty()
            val audioUrl = intent.getStringExtra(EXTRA_AI_AUDIO_URL)
            val payload = intent.getStringExtra("EXTRA_AI_PAYLOAD")

            // 只有有文字才記錄
            if (text.isNotBlank()) {
                appendTranscript(text, "ai")
            }

            if (!audioUrl.isNullOrBlank()) {
                // 有 URL（或本機檔路徑）就播放
                playTtsFromUrl(audioUrl, text)
            } else {
                // 沒音檔時才顯示文字；若正在播音就別覆蓋 UI
                if (text.isNotBlank() && !aiSpeaking) {
                    showAiAnswer(text)
                }
                if (!aiSpeaking) {                     // ★ 只在沒播音時才重啟 STT
                    uiHandler.postDelayed({ startSttIfPermitted() }, 200)
                }
            }
            Log.i(TAG_AI_REPLY, "text='${text.take(100)}' audioUrl=${audioUrl ?: "<none>"}")
        }
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

    // ===== STT 流程控制 =====
    private fun showMicListeningIcon() {
        if (!isVoiceRecording) {
            recordVoiceButton.setImageResource(R.drawable.ic_mic_registered)
        }
    }

    private fun startSttLoop() {
        if (sttLoopEnabled) return
        sttLoopEnabled = true
        sttLastPartial = ""
        sttEmitAllowed = true       // ← 迴圈啟動時就開門（一般情境要能出字）
        sttGateDeadline = 0L
        sttTriggeredByMonitor = false
        sttOneShot = false

        stopBackgroundVoiceMonitor()
        if (stt == null) initStt()
        updateStatus("辨識中…")

        sttHandler.post { startSttOnce() }
        sttHandler.removeCallbacks(sttWatchdog)
        sttHandler.postDelayed(sttWatchdog, 4000L)
    }

    private fun startSttIfPermitted() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED) {
            startSttLoop()              // ← 直接開 STT 迴圈
            showMicListeningIcon()
        } else {
            updateStatus("等待麥克風授權…")
        }
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

        sttHandler.post { startSttOnce() }
        sttHandler.removeCallbacks(sttWatchdog)
        sttHandler.postDelayed(sttWatchdog, 4000L)
    }

    private val sttWatchdog = object : Runnable {
        override fun run() {
            if (!sttLoopEnabled) return
            val now = System.currentTimeMillis()
            if (!isSttRunning || now - sttLastEventTs > 8000L) {
                try { stt?.cancel() } catch (_: Exception) {}
                startSttOnce()
            }
            sttHandler.postDelayed(this, 4000L)
        }
    }

    private fun startSttOnce() {
        if (!sttLoopEnabled || isSttRunning) return
        if (!hasAudioPermission()) {
            updateStatus("等待麥克風授權…")
            return
        }
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
        // 約 600ms 後依註冊狀態恢復背景監聽（不會搶麥）
        sttHandler.postDelayed({ setRegisteredUI(hasRegisteredElder()) }, 600L)
    }

    // ===== 麥克風搶占管理 =====
    private fun forceReleaseMic() {
        try { audioRecord?.stop() } catch (_: Exception) {}
        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null

        try { bgAudioRecord?.stop() } catch (_: Exception) {}
        try { bgAudioRecord?.release() } catch (_: Exception) {}
        bgAudioRecord = null

        monitorShouldRun = false
    }

    private fun pauseVoiceStuffForPlayback() {
        // 播放前停掉 STT 與背景監聽並釋放 MIC，避免搶資源
        pausedByPlayback = sttLoopEnabled
        if (sttLoopEnabled) stopSttLoop()
        stopBackgroundVoiceMonitor()
        forceReleaseMic()
    }

    private fun resumeVoiceStuffAfterPlayback() {
        // 播放結束後恢復原本語音狀態
        if (pausedByPlayback) startSttIfPermitted() else setRegisteredUI(hasRegisteredElder())
        pausedByPlayback = false
    }

    private fun placeServerBarBelowHud() {
        // 這兩個數字要和 addTextOverlay 裡的 textSize / pad 一致
        val hudTextSizePx = 32f
        val hudPadPx = 10f

        val p = Paint().apply { textSize = hudTextSizePx; isAntiAlias = true }
        val fm = p.fontMetrics
        val hudHeightPx = ((fm.bottom - fm.top) + 2f * hudPadPx).roundToInt()

        // 把「伺服器：」這條往下擺在 HUD 正下方，再加 8dp 的距離
        inferenceStatus.updateLayoutParams<FrameLayout.LayoutParams> {
            gravity = Gravity.TOP or Gravity.START
            topMargin = hudHeightPx + 8.dp()
            leftMargin = 5.dp() // 你原本的左邊距
        }
    }

    // ===== 背景長輩聲音監聽 =====
    private fun startBackgroundVoiceMonitor() {
        if (monitorShouldRun) return
        monitorShouldRun = true
        monitorThread = thread(start = true, isDaemon = true) {
            val sampleRate = 16000
            val frameMs = 120L
            val silenceNeedFrames = 3
            val hangoverFrames = 1
            val nonElderEndFrames = 4
            val silentDbThreshold = -48f
            val maxSegFrames = 120
            val preFrames = 6
            val startStreakNeed = 1
            val unlockStreakNeed = 2
            val minElderFramesToUnlock = 5

            var silentFrames = 0
            var nonElderFrames = 0
            var nonSpeechFrames = 0
            var segFrames = 0
            var elderStreak = 0
            var elderFramesInSeg = 0
            var tailFrames = 0
            var sttRequested = false

            /*
            val frameMs = 120L
            val silenceNeedFrames = 3
            val hangoverFrames = 2
            val nonElderEndFrames = 8
            val silentDbThreshold = -48f
            val maxSegFrames = 120
            val preFrames = 10
            val startStreakNeed = 1
            val unlockStreakNeed = 2
            val minElderFramesToUnlock = 4

            var silentFrames = 0
            var nonElderFrames = 0
            var nonSpeechFrames = 0
            var segFrames = 0
            var elderStreak = 0
            var elderFramesInSeg = 0
            var tailFrames = 0
            var sttRequested = false
             */

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
                return audible && elderFramesInSeg >= 6 && ratio >= 0.8f
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

                            lastElderAt = System.currentTimeMillis()

                            if (segFrames == 0) {
                                if (elderStreak >= startStreakNeed) {
                                    for (pr in pre) currentSegment.add(pr)
                                    currentSegment.add(pcm)
                                    segFrames++
                                    elderFramesInSeg++
                                }
                            } else {
                                currentSegment.add(pcm)
                                segFrames++
                                elderFramesInSeg++
                            }

                            // 更嚴格解鎖：連續幀 + 累積幀 + 無外放
                            if (!sttEmitAllowed &&
                                elderStreak >= unlockStreakNeed &&
                                elderFramesInSeg >= minElderFramesToUnlock &&
                                !aiSpeaking
                            ) {
                                val justUnlocked = !sttEmitAllowed
                                sttEmitAllowed = true
                                Log.d("VoiceMonitor", "解鎖輸出（elderStreak=$elderStreak, elderFramesInSeg=$elderFramesInSeg）")

                                if (justUnlocked && sttLastPartial.isNotBlank()) {
                                    broadcastStt(sttLastPartial, true)
                                    appendTranscript(sttLastPartial, "partial")
                                }

                                if (!sttRequested) {
                                    sttRequested = true
                                    runOnUiThread {
                                        stopBackgroundVoiceMonitor()
                                        sttHandler.postDelayed({ startSttFromMonitor() }, 40)
                                    }
                                }
                            } else if (!sttEmitAllowed &&
                                elderStreak >= unlockStreakNeed &&
                                elderFramesInSeg >= minElderFramesToUnlock &&
                                isDevicePlaying()) {
                                Log.d("VoiceMonitor", "⏸ 裝置正在播放，暫不解鎖")
                            }
                        } else {
                            updateStatus("不是長輩聲音")
                            elderStreak = 0
                            nonElderFrames++

                            if (segFrames > 0) {
                                if (tailFrames < hangoverFrames) {
                                    currentSegment.add(pcm)
                                    segFrames++
                                    tailFrames++
                                }
                            }

                            // 連續非長輩幀達門檻 → 關閘避免外部語音接手
                            if (sttEmitAllowed && nonElderFrames >= 4) {
                                sttEmitAllowed = false
                                Log.d("VoiceMonitor", "關閉輸出（nonElderFrames=$nonElderFrames）")
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

    private fun hasMicPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this,
            android.Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

    private fun ensureMicPermissionOrAsk(): Boolean {
        val ok = hasMicPermission()
        if (!ok) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                MIC_PERMISSION_CODE
            )
        }
        return ok
    }

    private fun ensureMonitorRecorder(sampleRate: Int): AudioRecord? {
        // 先確認/請求權限
        if (!ensureMicPermissionOrAsk()) return null

        var r = bgAudioRecord
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (r == null || r.state != AudioRecord.STATE_INITIALIZED) {
            val bufSize = maxOf(minBuf, sampleRate / 5 * 2) // ≈200ms buffer

            val tmp = try {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufSize
                )
            } catch (se: SecurityException) {
                Log.e("VoiceMonitor", "AudioRecord ctor SecurityException: ${se.message}")
                return null
            } catch (e: Exception) {
                Log.e("VoiceMonitor", "AudioRecord ctor failed: ${e.message}")
                return null
            }

            if (tmp.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("VoiceMonitor", "AudioRecord init failed (state=${tmp.state}). Mic busy?")
                try { tmp.release() } catch (_: Exception) {}
                return null
            }

            try {
                tmp.startRecording()
            } catch (se: SecurityException) {
                Log.e("VoiceMonitor", "startRecording SecurityException: ${se.message}")
                try { tmp.release() } catch (_: Exception) {}
                return null
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
            } catch (se: SecurityException) {
                Log.e("VoiceMonitor", "startRecording() SecurityException on existing recorder: ${se.message}")
                try { r.release() } catch (_: Exception) {}
                bgAudioRecord = null
                return null
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

    // ===== 音訊工具 =====
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

    // ===== 註冊/重置（錄音 + 播放 + 確認 + 儲存） =====
    // REPLACE THIS WHOLE FUNCTION
    private fun recordAndShowDialog(onFinish: () -> Unit) {
        // 停 STT / 背景監聽，釋放麥克風，避免資源互搶
        if (sttLoopEnabled) stopSttLoop() else stopBackgroundVoiceMonitor()
        forceReleaseMic()

        if (!hasAudioPermission()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
            Toast.makeText(this, "請先允許麥克風權限再進行錄音", Toast.LENGTH_SHORT).show()
            return
        }

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

        try {
            try {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED
                ) {
                    audioRecord?.startRecording()
                    runOnUiThread { updateStatus("錄音中…") }
                } else {
                    Toast.makeText(this, "請先允許麥克風權限", Toast.LENGTH_SHORT).show()
                    return
                }
            } catch (e: SecurityException) {
                Log.e("AudioRecord", "startRecording failed: ${e.message}")
                Toast.makeText(this, "錄音失敗：${e.message}", Toast.LENGTH_SHORT).show()
                return
            }

            runOnUiThread {
                isVoiceRecording = true
                updateStatus("錄音中…")
                // 如需改圖示/底色，可在這裡做（避免用到你沒有的 drawable）
                // recordVoiceButton.setImageResource(R.drawable.ic_stop_white)
            }
        } catch (e: SecurityException) {
            Log.e("AudioRecord", "startRecording failed: ${e.message}")
        }

        // 固定時長錄音（無手動停止）
        thread {
            val durationMillis = 10_000L
            val startTime = System.currentTimeMillis()

            while (System.currentTimeMillis() - startTime < durationMillis) {
                val readBytes = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (readBytes > 0) pcmData.write(buffer, 0, readBytes)
            }

            try { audioRecord?.stop() } catch (_: Exception) {}
            try { audioRecord?.release() } catch (_: Exception) {}
            audioRecord = null

            val wavFile = File(getExternalFilesDir(null), "elder_sample.wav")
            saveAsWavFile(pcmData.toByteArray(), wavFile, sampleRate, 1, 16)

            runOnUiThread {
                isVoiceRecording = false
                onFinish()
                showConfirmDialog(wavFile)
            }
        }
    }


    private fun requestPlaybackFocus(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val attrs = android.media.AudioAttributes.Builder()
                .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            val req = android.media.AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
                .setAudioAttributes(attrs)
                .setOnAudioFocusChangeListener { /* no-op */ }
                .build()
            playbackFocusRequest = req
            audioManager.requestAudioFocus(req) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(
                null, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
            ) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        }
    }

    private fun abandonPlaybackFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            playbackFocusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
        } else {
            @Suppress("DEPRECATION")
            audioManager.abandonAudioFocus(null)
        }
        playbackFocusRequest = null
    }

    // 錄音流程的 UI 控制（不再用深灰底）
    private fun setRegisteringUI(registering: Boolean) {
        if (registering) {
            // 顯示狀態
            updateStatus("註冊中…")
            // 鎖住按鈕，避免連點
            recordVoiceButton.isEnabled = false
            // 確保沒有任何背景（拔掉深灰底）
            recordVoiceButton.background = null
            recordVoiceButton.setBackgroundResource(0)
        } else {
            // 錄音結束 → 回到一般狀態（依是否已註冊）
            recordVoiceButton.isEnabled = true
            recordVoiceButton.background = null
            recordVoiceButton.setBackgroundResource(0)
            // 可改成你要的提示文字
            updateStatus(if (hasRegisteredElder()) "已註冊，待機中" else "請先註冊聲音")
        }
    }

    // REPLACE THIS WHOLE FUNCTION
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
            val wasSttRunning = sttLoopEnabled
            var resumeAllowed = true   // ← 重新錄製時設為 false，避免恢復 STT

            fun stopPlayer() {
                runCatching { player?.setOnCompletionListener(null) }
                runCatching { player?.stop() }
                runCatching { player?.release() }
                player = null

                // 只有在允許時才恢復 STT/背景監聽（重新錄製時不恢復）
                if (resumeAllowed) {
                    if (wasSttRunning) {
                        sttHandler.postDelayed({ startSttIfPermitted() }, 150)
                    } else {
                        setRegisteredUI(hasRegisteredElder())
                    }
                }
            }

            dialog.setOnDismissListener { stopPlayer() }

            // 播放
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    stopPlayer()
                    // 播放前先確保不會被 STT/監聽佔用麥克風
                    if (sttLoopEnabled) stopSttLoop() else stopBackgroundVoiceMonitor()
                    forceReleaseMic()

                    player = MediaPlayer().apply {
                        setDataSource(wavFile.absolutePath)
                        setOnPreparedListener { mp -> mp.start() }
                        setOnCompletionListener {
                            Toast.makeText(
                                this@MainActivity,
                                "播放結束，可按「確認儲存」或「重新錄製」",
                                Toast.LENGTH_SHORT
                            ).show()
                            stopPlayer()
                        }
                        setOnErrorListener { _, what, extra ->
                            Toast.makeText(
                                this@MainActivity,
                                "播放失敗（$what/$extra）",
                                Toast.LENGTH_SHORT
                            ).show()
                            stopPlayer()
                            true
                        }
                        prepareAsync()
                    }
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, "播放失敗：${e.message}", Toast.LENGTH_SHORT).show()
                    stopPlayer()
                }
            }

            // 重新錄製
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                resumeAllowed = false   // ← 關掉自動恢復 STT
                stopPlayer()
                dialog.dismiss()

                // 立即切回錄製流程（固定時長錄完）
                isVoiceRecording = true
                updateStatus("註冊中…")
                recordAndShowDialog {
                    isVoiceRecording = false
                    updateStatus("錄音完成，請確認聲音樣本")
                }
            }

            // 確認儲存
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                stopPlayer()  // 儲存前確保已停止播放
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
                            startSttIfPermitted()
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

    // ===== WAV 存檔 =====
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

    // ===== 聲紋狀態 =====
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
        if (isVoiceRecording) return

        if (registered) {
            recordVoiceButton.setImageResource(R.drawable.ic_mic_registered)
            recordVoiceButton.setBackgroundResource(android.R.color.transparent)
            recordVoiceButton.background = null
            recordVoiceButton.isEnabled = true

            recordVoiceButton.isLongClickable = true
            recordVoiceButton.setOnLongClickListener {
                AlertDialog.Builder(this)
                    .setTitle("重置長輩聲音")
                    .setMessage("確定要清除已註冊的聲紋嗎？")
                    .setPositiveButton("清除") { _, _ -> resetRegistration() }
                    .setNegativeButton("取消", null)
                    .show()
                true
            }

            if (sttLoopEnabled) {
                updateStatus("辨識中…")
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

            recordVoiceButton.isLongClickable = false
            recordVoiceButton.setOnLongClickListener(null)
        }
    }

    // 一鍵重置：清除已註冊聲紋與狀態
    private fun resetRegistration() {
        try {
            val f1 = File(getExternalFilesDir(null), "elder_embedding.vec")
            if (f1.exists()) f1.delete()
        } catch (e: Exception) {
            Log.e("CameraDetectActivity", "刪除外部聲紋檔案失敗", e)
        }
        try {
            val f2 = File(filesDir, "elder_embedding.vec")
            if (f2.exists()) f2.delete()
        } catch (e: Exception) {
            Log.e("CameraDetectActivity", "刪除內部聲紋檔案失敗", e)
        }

        stopBackgroundVoiceMonitor()
        previousEmbedding = null
        currentSegment.clear()
        setRegisteredUI(false)
        Toast.makeText(this, "已重置為未註冊狀態", Toast.LENGTH_SHORT).show()
    }

    override fun onResume() {
        super.onResume()
        // 恢復註冊狀態（會自動決定是否開背景監聽）
        setRegisteredUI(hasRegisteredElder())

        // 有麥克風權限，且目前沒在跑 → 啟動 STT
        if (hasAudioPermission() && !sttLoopEnabled && !isSttRunning) {
            startSttIfPermitted()
        }
    }

    override fun onPause() {
        super.onPause()

        // === 暫停影像擷取（保險）===
        allowProcess = false
        lastProcessTimeMs = 0L

        // === 暫停語音相關 ===
        sttEmitAllowed = false
        sttGateDeadline = 0L

        runCatching { stt?.stopListening() }
        runCatching { stt?.cancel() }
        if (sttLoopEnabled) stopSttLoop()

        stopBackgroundVoiceMonitor()

        // === 關閉 WebSocket，並停自動重連 ===
        // 1) 真正關連線（WsManager.close() 會設 manualClose=true / wantReconnect=false）
        try { ws.close(1000, "paused") } catch (_: Exception) {}
        // 2) 更新內部旗標與 UI（避免回到前景又自動連）
        sendWsEnabled = false
        // 切 UI 狀態到「關」；這行會觸發你的 listener 走到 ws.close()，但我們已先 close 所以 OK
        if (switchSendWs.isChecked) switchSendWs.isChecked = false
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        imageAnalysis?.clearAnalyzer()
        cameraExecutor.shutdown()
        // === 語音 ===
        try { stt?.destroy() } catch (_: Exception) {}
        stt = null
        stopBackgroundVoiceMonitor()
        try { audioRecord?.stop() } catch (_: Exception) {}
        try { audioRecord?.release() } catch (_: Exception) {}
        audioRecord = null
        try { bgAudioRecord?.stop() } catch (_: Exception) {}
        try { bgAudioRecord?.release() } catch (_: Exception) {}
        bgAudioRecord = null
    }

    private fun readRawModel(resId: Int): ByteArray =
        resources.openRawResource(resId).use { it.readBytes() }

    private fun readClasses(): List<String> =
        resources.openRawResource(R.raw.classes).bufferedReader().readLines()

    /** 將「原圖座標」轉成 overlay 畫面座標（與 PreviewView: fitCenter 一致） */
    private fun calcFitCenter(vw: Int, vh: Int, srcW: Int, srcH: Int): Triple<Float, Float, Float> {
        val r = min(vw / srcW.toFloat(), vh / srcH.toFloat())
        val padX = (vw - srcW * r) / 2f
        val padY = (vh - srcH * r) / 2f
        return Triple(r, padX, padY)
    }

    /** 畫物件框 */
    private fun drawDetectionsOnOverlay(view: View, srcW: Int, srcH: Int, boxes: Array<FloatArray>) {
        if (!renderEnabled) return
        val vw = view.width.coerceAtLeast(1)
        val vh = view.height.coerceAtLeast(1)
        val (r, padX, padY) = calcFitCenter(vw, vh, srcW, srcH)

        val d = object : android.graphics.drawable.Drawable() {
            private val boxPaint = android.graphics.Paint().apply {
                style = android.graphics.Paint.Style.STROKE; strokeWidth = 4f
                color = android.graphics.Color.GREEN; isAntiAlias = true
            }
            private val textPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.WHITE; textSize = 28f; isAntiAlias = true
            }
            private val bgPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.argb(160, 0, 0, 0)
            }
            override fun draw(canvas: android.graphics.Canvas) {
                for (b in boxes) {
                    val cx = b[0] * r + padX; val cy = b[1] * r + padY
                    val w = b[2] * r; val h = b[3] * r
                    val left = cx - w/2f; val top = cy - h/2f
                    val right = cx + w/2f; val bottom = cy + h/2f
                    canvas.drawRect(left, top, right, bottom, boxPaint)

                    val clsId = b[5].toInt()
                    val name = if (clsId in 0 until classes.size) classes[clsId] else clsId.toString()
                    val label = "$name:${"%.2f".format(b[4])}"
                    val pad = 6f
                    val tw = textPaint.measureText(label)
                    val fm = textPaint.fontMetrics
                    val th = fm.bottom - fm.top
                    var bgTop = top - th - 2*pad
                    var bgBottom = top
                    if (bgTop < 0) { bgTop = top; bgBottom = top + th + 2*pad }
                    canvas.drawRect(left, bgTop, left + tw + 2*pad, bgBottom, bgPaint)
                    canvas.drawText(label, left + pad, bgBottom - pad - fm.bottom, textPaint)
                }
            }
            override fun setAlpha(alpha: Int) {}
            override fun setColorFilter(colorFilter: android.graphics.ColorFilter?) {}
            @Deprecated("Deprecated in Java")
            override fun getOpacity(): Int = android.graphics.PixelFormat.TRANSLUCENT
        }.apply { setBounds(0, 0, vw, vh) }

        view.overlay.add(d)
    }

    /** 畫骨架（盒+關節+連線） */
    private fun drawPoseOnOverlay(view: View, srcW: Int, srcH: Int, poses: List<Pose>) {
        if (!renderEnabled) return
        val vw = view.width.coerceAtLeast(1)
        val vh = view.height.coerceAtLeast(1)
        val (r, padX, padY) = calcFitCenter(vw, vh, srcW, srcH)

        val edges = arrayOf(
            intArrayOf(5,6), intArrayOf(5,7), intArrayOf(7,9),
            intArrayOf(6,8), intArrayOf(8,10), intArrayOf(5,11),
            intArrayOf(6,12), intArrayOf(11,12), intArrayOf(11,13),
            intArrayOf(13,15), intArrayOf(12,14), intArrayOf(14,16),
            intArrayOf(0,5), intArrayOf(0,6), intArrayOf(0,1),
            intArrayOf(0,2), intArrayOf(1,3), intArrayOf(2,4)
        )

        val d = object : android.graphics.drawable.Drawable() {
            private val kpPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.CYAN; style = android.graphics.Paint.Style.FILL; isAntiAlias = true
            }
            private val linePaint = android.graphics.Paint().apply {
                color = android.graphics.Color.GREEN; strokeWidth = 4f; style = android.graphics.Paint.Style.STROKE; isAntiAlias = true
            }
            private val boxPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.MAGENTA; strokeWidth = 3f; style = android.graphics.Paint.Style.STROKE; isAntiAlias = true
            }
            override fun draw(canvas: android.graphics.Canvas) {
                for (p in poses) {
                    val cx = p.box[0] * r + padX; val cy = p.box[1] * r + padY
                    val w  = p.box[2] * r;       val h  = p.box[3] * r
                    val l = cx - w/2f; val t = cy - h/2f; val rt = cx + w/2f; val b = cy + h/2f
                    canvas.drawRect(l, t, rt, b, boxPaint)

                    for (e in edges) {
                        val a = p.keypoints[e[0]]
                        val bpt = p.keypoints[e[1]]
                        if (a[2] > 0.5f && bpt[2] > 0.5f) {
                            canvas.drawLine(a[0]*r+padX, a[1]*r+padY, bpt[0]*r+padX, bpt[1]*r+padY, linePaint)
                        }
                    }
                    for (kp in p.keypoints) {
                        if (kp[2] > 0.5f) canvas.drawCircle(kp[0]*r+padX, kp[1]*r+padY, 4f, kpPaint)
                    }
                }
            }
            override fun setAlpha(alpha: Int) {}
            override fun setColorFilter(colorFilter: android.graphics.ColorFilter?) {}
            @Deprecated("Deprecated in Java")
            override fun getOpacity(): Int = android.graphics.PixelFormat.TRANSLUCENT
        }.apply { setBounds(0, 0, vw, vh) }

        view.overlay.add(d)
    }
    override fun onStart() {
        super.onStart()

        if (!aiReceiverRegistered) {
            val filter = IntentFilter(ACTION_AI_REPLY)
            ContextCompat.registerReceiver(
                /* context = */ this,
                /* receiver = */ aiReplyReceiver,
                /* filter = */ filter,
                /* flags = */ ContextCompat.RECEIVER_NOT_EXPORTED
            )
            aiReceiverRegistered = true
        }
    }

    override fun onStop() {
        super.onStop()
        if (aiReceiverRegistered) {
            try { unregisterReceiver(aiReplyReceiver) } catch (_: Exception) {}
            aiReceiverRegistered = false
        }
    }

    private fun safeRotation(): Int {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            this.display?.rotation?.let { return it }
        }
        @Suppress("DEPRECATION")
        try { return windowManager.defaultDisplay.rotation } catch (_: Throwable) {}
        previewView.display?.rotation?.let { return it }
        return Surface.ROTATION_0
    }

    private fun addTextOverlay(view: View, text: String) {
        if (!renderEnabled) return
        val vw = view.width.coerceAtLeast(1)
        val vh = view.height.coerceAtLeast(1)
        val d = object : android.graphics.drawable.Drawable() {
            private val pad = 10f
            private val textPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.WHITE
                textSize = 32f
                isAntiAlias = true
            }
            private val bgPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.argb(150, 0, 0, 0)
                isAntiAlias = true
            }
            override fun draw(canvas: android.graphics.Canvas) {
                val fm = textPaint.fontMetrics
                val th = fm.bottom - fm.top
                val tw = textPaint.measureText(text)
                val left = pad
                val top = pad
                canvas.drawRect(left - pad, top - th - pad, left + tw + pad, top + pad, bgPaint)
                canvas.drawText(text, left, top - fm.top, textPaint)
            }
            override fun setAlpha(alpha: Int) {}
            override fun setColorFilter(colorFilter: android.graphics.ColorFilter?) {}
            override fun getOpacity() = android.graphics.PixelFormat.TRANSLUCENT
        }.apply { setBounds(0, 0, vw, vh) }
        view.overlay.add(d)
    }

    private fun processFrame(bitmap: Bitmap) {
        var objMs = -1L
        var poseMs = -1L
        val now = SystemClock.elapsedRealtime()

        // 只在到時間時才跑 POSE
        if (poseDetectEnabled && now - lastPoseInferAt >= POSE_INFER_MS) {
            val t0 = now
            val poseRes = poseDetector.detect(bitmap, ortEnv, poseSession)
            poseMs = SystemClock.elapsedRealtime() - t0
            lastPoseInferAt = SystemClock.elapsedRealtime()
            lastPoseResult = poseRes
        }

        // 只在到時間時才跑 DETECT
        if (objectDetectEnabled && now - lastDetectInferAt >= DETECT_INFER_MS) {
            val t0 = SystemClock.elapsedRealtime()
            val objRes = objDetector.detect(bitmap, ortEnv, objSession)
            objMs = SystemClock.elapsedRealtime() - t0
            lastDetectInferAt = SystemClock.elapsedRealtime()
            lastObjResult = objRes
        }

        frameId = (frameId % 60) + 1

        // —— Overlay 與 HUD（用快取畫） —— //
        val objResForDraw = lastObjResult
        val poseResForDraw = lastPoseResult

        runOnUiThread {
            overlayView.overlay.clear()
            objResForDraw?.let { drawDetectionsOnOverlay(overlayView, bitmap.width, bitmap.height, it.outputBox) }
            poseResForDraw?.let { drawPoseOnOverlay(overlayView, bitmap.width, bitmap.height, it.poses) }

            val hud = buildString {
                if (objectDetectEnabled) append("OBJ:${objResForDraw?.outputBox?.size ?: 0} ${if (objMs>=0) "${objMs}ms" else ""}   ")
                if (poseDetectEnabled)   append("POSE:${poseResForDraw?.poses?.size ?: 0} ${if (poseMs>=0) "${poseMs}ms" else ""}")
            }.trim()
            if (hud.isNotEmpty()) addTextOverlay(overlayView, hud)

            if (!renderEnabled) overlayView.overlay.clear()
        }
    }

    private fun startSenderLoop() {
        poseBurstCounter = 0
        lastPoseSendAt = 0L
        lastDetectSendAt = 0L

        if (senderHandler != null) return

        senderThread = android.os.HandlerThread("ws-sender").apply { start() }
        senderHandler = Handler(senderThread!!.looper)

        senderRunnable = object : Runnable {
            override fun run() {
                try {
                    if (!sendWsEnabled) return
                    val now = SystemClock.elapsedRealtime()
                    val poseInterval = 200L     // 5 fps
                    val detectInterval = 1000L  // 1 fps

                    val wantPose = poseDetectEnabled
                    val wantDetect = objectDetectEnabled

                    // 每 200ms 送一包 Pose；每第 5 包（約每 1s）再夾 Detect
                    if (wantPose && (now - lastPoseSendAt >= poseInterval)) {
                        poseBurstCounter++

                        val includePose = true
                        val includeDetect = wantDetect &&
                                (poseBurstCounter % 5 == 0) &&         // 第 5、10、15... 包
                                (now - lastDetectSendAt >= detectInterval)

                        val w = if (lastFrameW > 0) lastFrameW else 640
                        val h = if (lastFrameH > 0) lastFrameH else 480

                        if (includePose || includeDetect) {
                            val frameJson = buildFrameJson(
                                fid = frameId,
                                width = w,
                                height = h,
                                obj = if (includeDetect) lastObjResult else null,
                                pose = if (includePose)   lastPoseResult else null,
                                includeDetect = includeDetect,
                                includePose   = includePose
                            ).toString()
                            try { ws.send(frameJson) } catch (t: Throwable) { Log.e(TAG, "WS send error", t) }
                            lastPoseSendAt = now
                            if (includeDetect) lastDetectSendAt = now
                        }
                    } else if (!wantPose && wantDetect && (now - lastDetectSendAt >= detectInterval)) {
                        // 只開 Detect 的備援路徑：每 1s 單獨送 Detect
                        val w = if (lastFrameW > 0) lastFrameW else 640
                        val h = if (lastFrameH > 0) lastFrameH else 480
                        val frameJson = buildFrameJson(
                            fid = frameId, width = w, height = h,
                            obj = lastObjResult, pose = null,
                            includeDetect = true, includePose = false
                        ).toString()
                        try { ws.send(frameJson) } catch (t: Throwable) { Log.e(TAG, "WS send error", t) }
                        lastDetectSendAt = now
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "sender loop crashed", t)
                } finally {
                    senderHandler?.postDelayed(this, 200L) // 200ms 一拍
                }
            }
        }
        senderHandler?.postDelayed(senderRunnable!!, 200L)
    }

    private fun stopSenderLoop() {
        senderHandler?.removeCallbacksAndMessages(null)
        senderHandler = null

        senderThread?.quitSafely()
        senderThread = null
        senderRunnable = null
    }

    private fun handleServerMessage(msg: String) {
        try {
            val j = JSONObject(msg)
            if (j.optString("type") != "inference") return

            val multi = j.optJSONObject("multi")
            val bin   = j.optJSONObject("binary")

            val multiPred = multi?.optString("pred").orEmpty()
            val multiIdx  = multi?.optInt("pred_idx", -1) ?: -1
            val multiProbs = multi?.optJSONArray("probs")
            val multiPct = if (multiIdx >= 0 && multiProbs != null && multiIdx < multiProbs.length())
                (multiProbs.optDouble(multiIdx, 0.0) * 100.0).toInt() else -1

            val binPred = bin?.optString("pred").orEmpty()             // "fall" 或 "non_fall"
            val binProbs = bin?.optJSONArray("probs")
            val binFallProb = binProbs?.optDouble(1, Double.NaN) ?: Double.NaN // probs[1] = fall 機率
            val binThr = bin?.optDouble("thr", 0.65) ?: 0.65

            val fallAlert = pushFall((binPred == "fall") && !binFallProb.isNaN() && binFallProb >= binThr)

            runOnUiThread {
                inferenceStatus.text = if (fallAlert) {
                    "⚠️ 跌倒 ${(binFallProb * 100).toInt()}%"
                } else {
                    if (multiPct >= 0) "姿態：$multiPred ${multiPct}%"
                    else "姿態：$multiPred"
                }
            }
        } catch (_: Exception) {
            // 非 JSON 就忽略或簡短顯示
            runOnUiThread { /* inferenceStatus.text = "推論：" */ }
        }
    }

    private val fallQueue: ArrayDeque<Boolean> = ArrayDeque(3)
    private fun pushFall(b: Boolean): Boolean {
        if (fallQueue.size == 3) fallQueue.removeFirst()
        fallQueue.addLast(b)
        return fallQueue.all { it }
    }

    // 取代或並存於 buildObjectJson / buildPoseJson 旁邊
    private fun buildFrameJson(
        fid: Long,
        width: Int,
        height: Int,
        obj: ObjectResult?,
        pose: PoseResult?,
        includeDetect: Boolean = true,
        includePose: Boolean = true
    ): JSONObject {
        val root = JSONObject()
        root.put("type", "frame")
        root.put("frame_id", fid)
        root.put("timestamp_ms", System.currentTimeMillis())
        root.put("image_size", JSONObject().apply {
            put("width", width)
            put("height", height)
        })

        // detect
        if (includeDetect) {
            val detectArr = JSONArray()
            obj?.outputBox?.forEach { b ->
                val clsId = b[5].toInt()
                val name = if (clsId in 0 until classes.size) classes[clsId] else clsId.toString()
                detectArr.put(JSONObject().apply {
                    put("cls_id", clsId)
                    put("cls_name", name)
                    put("score", b[4].toDouble())
                    put("bbox", JSONObject().apply {
                        put("cx", b[0].toDouble()); put("cy", b[1].toDouble())
                        put("w",  b[2].toDouble()); put("h",  b[3].toDouble())
                    })
                })
            }
            root.put("detect", detectArr)
        }

        // pose
        if (includePose) {
            val poseArr = JSONArray()
            pose?.poses?.forEach { p ->
                val person = JSONObject().apply {
                    put("score", p.score.toDouble())
                    put("bbox", JSONObject().apply {
                        put("cx", p.box[0].toDouble()); put("cy", p.box[1].toDouble())
                        put("w",  p.box[2].toDouble()); put("h",  p.box[3].toDouble())
                    })
                    val kps = JSONArray()
                    for (kp in p.keypoints) {
                        kps.put(JSONObject().apply {
                            put("x", kp[0].toDouble())
                            put("y", kp[1].toDouble())
                            put("confidence", kp[2].toDouble())
                        })
                    }
                    put("keypoints", kps)
                }
                poseArr.put(person)
            }
            root.put("pose", poseArr)
        }
        return root
    }

    // ★ 新增：建立物件偵測 JSON（與單張輸出對齊：使用 classes 名稱）
    private fun buildObjectJson(fid: Long, bmp: Bitmap, res: ObjectResult?): JSONObject {
        val root = JSONObject()
        root.put("type", "object")
        root.put("frame_id", fid)
        root.put("timestamp_ms", System.currentTimeMillis())
        root.put("image_size", JSONObject().apply {
            put("width", bmp.width); put("height", bmp.height)
        })

        val arr = JSONArray()
        res?.outputBox?.forEach { b ->
            val clsId = b[5].toInt()
            val name = if (clsId in 0 until classes.size) classes[clsId] else clsId.toString()
            arr.put(JSONObject().apply {
                put("cls_id", clsId)
                put("cls_name", name)
                put("score", b[4].toDouble())
                put("bbox", JSONObject().apply {
                    put("cx", b[0].toDouble()); put("cy", b[1].toDouble())
                    put("w",  b[2].toDouble()); put("h",  b[3].toDouble())
                })
            })
        }
        root.put("detect", arr)
        return root
    }

    // ★ 新增：建立骨架偵測 JSON
    private fun buildPoseJson(fid: Long, bmp: Bitmap, res: PoseResult?): JSONObject {
        val root = JSONObject()
        root.put("type", "pose")
        root.put("frame_id", fid)
        root.put("timestamp_ms", System.currentTimeMillis())
        root.put("image_size", JSONObject().apply {
            put("width", bmp.width); put("height", bmp.height)
        })

        val persons = JSONArray()
        res?.poses?.forEach { p ->
            val person = JSONObject().apply {
                put("score", p.score.toDouble())
                put("bbox", JSONObject().apply {
                    put("cx", p.box[0].toDouble()); put("cy", p.box[1].toDouble())
                    put("w",  p.box[2].toDouble()); put("h",  p.box[3].toDouble())
                })

                val kps = JSONArray()
                for (kp in p.keypoints) {
                    kps.put(JSONObject().apply {
                        put("x", kp[0].toDouble())
                        put("y", kp[1].toDouble())
                        put("confidence", kp[2].toDouble())  // ← 改這裡
                    })
                }
                put("keypoints", kps)
            }
            persons.put(person)
        }
        root.put("pose", persons)
        return root
    }
}
