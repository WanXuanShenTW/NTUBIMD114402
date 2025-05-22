package com.example.myapplication

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.video.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var recordIndicator: View
    private lateinit var headerOverlay: View
    private lateinit var backButton: ImageView
    private lateinit var cameraExecutor: ExecutorService
    private var blinkAnimator: ObjectAnimator? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var isOverlayVisible = false
    private var isRecording = false
    private var segmentCount = 0  // ✅ 新增變數：錄幾段了

    private val REQUEST_CODE_PERMISSIONS = 1001
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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
                segmentCount = 0
                startRecording()
            } catch (e: Exception) {
                Log.e("CameraX", "啟動相機失敗", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startBlinkingIndicator() {
        runOnUiThread {
            if (!isRecording) return@runOnUiThread
            blinkAnimator?.cancel()
            recordIndicator.alpha = 1f
            recordIndicator.visibility = View.VISIBLE
            blinkAnimator = ObjectAnimator.ofFloat(recordIndicator, View.ALPHA, 1f, 0f).apply {
                duration = 500
                repeatMode = ValueAnimator.REVERSE
                repeatCount = ValueAnimator.INFINITE
                start()
            }
        }
    }

    private fun stopBlinkingIndicator() {
        runOnUiThread {
            blinkAnimator?.cancel()
            recordIndicator.clearAnimation()
            recordIndicator.alpha = 1f
            recordIndicator.visibility = View.GONE
        }
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return
        if (segmentCount >= 3) return  // 最多錄 3 段

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val videoFile = File(cacheDir, "video_$timestamp.mp4")
        val outputOptions = FileOutputOptions.Builder(videoFile).build()

        try {
            recording = videoCapture.output
                .prepareRecording(this, outputOptions)
                .withAudioEnabled()
                .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                    when (recordEvent) {
                        is VideoRecordEvent.Start -> {
                            isRecording = true
                            startBlinkingIndicator()
                            Toast.makeText(this, "開始錄影（第 ${segmentCount + 1} 段）", Toast.LENGTH_SHORT).show()
                        }
                        is VideoRecordEvent.Finalize -> {
                            isRecording = false
                            stopBlinkingIndicator()
                            if (recordEvent.hasError()) {
                                Log.e("VideoCapture", "錄影錯誤：${recordEvent.error}")
                                startRecording()
                            } else {
                                uploadVideo(videoFile) {
                                    segmentCount++
                                    startRecording()
                                }
                            }
                        }
                    }
                }
        } catch (e: SecurityException) {
            Toast.makeText(this, "缺少錄音或相機權限", Toast.LENGTH_SHORT).show()
            Log.e("VideoCapture", "權限錯誤：${e.message}")
        }

        cameraExecutor.execute {
            Thread.sleep(3000)  // ✅ 只錄 1 秒
            runOnUiThread {
                Toast.makeText(this, "正在儲存影片...", Toast.LENGTH_SHORT).show()
            }
            recording?.stop()
        }
    }

    private fun uploadVideo(videoFile: File, onComplete: () -> Unit) {
        val userId = "1"
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("id", userId)
            .addFormDataPart("video", videoFile.name, videoFile.asRequestBody("video/mp4".toMediaTypeOrNull()))
            .build()

        val request = Request.Builder()
            .url("https://eb2e-150-116-79-4.ngrok-free.app/detect_fall_video")
            .post(requestBody)
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("UploadVideo", "連線失敗：${e.message}", e)  // 🔍顯示詳細錯誤
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "❌ 上傳失敗：${e.message}", Toast.LENGTH_LONG).show()
                    onComplete()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val bodyString = response.body?.string()
                Log.d("UploadVideo", "伺服器回應（${response.code}）：$bodyString")

                runOnUiThread {
                    if (response.isSuccessful) {
                        Toast.makeText(this@MainActivity, "✅ 上傳成功：$bodyString", Toast.LENGTH_LONG).show()
                        videoFile.delete()
                    } else {
                        Toast.makeText(this@MainActivity, "⚠️ 錯誤碼 ${response.code}：$bodyString", Toast.LENGTH_LONG).show()
                    }
                    onComplete()
                }
            }
        })
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "需要相機與錄音權限", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        recording?.close()
        recording = null
        blinkAnimator?.cancel()
    }
}
