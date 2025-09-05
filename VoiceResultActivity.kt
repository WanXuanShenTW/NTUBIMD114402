package com.example.myapplication

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import androidx.core.content.ContextCompat
import android.widget.ImageView
import android.widget.LinearLayout


class VoiceResultActivity : AppCompatActivity() {

    private lateinit var tvLog: TextView
    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    private val sttReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != MainActivity.ACTION_STT_UPDATE) return
            val text = intent.getStringExtra(MainActivity.EXTRA_STT_TEXT).orEmpty()
            val isPartial = intent.getBooleanExtra(MainActivity.EXTRA_STT_IS_PARTIAL, false)
            if (text.isBlank() || isPartial) return

            val ts = timeFmt.format(Date())
            val line = "[$ts] $text"
            appendLine(line)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_result)

        tvLog = findViewById(R.id.tvLog)
        tvLog.movementMethod = ScrollingMovementMethod()

        loadTranscript() // 載入歷史（包含 partial 與 final）

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter(MainActivity.ACTION_STT_UPDATE)

        ContextCompat.registerReceiver(
            this,
            sttReceiver,
            filter,
            /* broadcastPermission = */ null,
            /* scheduler = */ null,
            /* flags = */ ContextCompat.RECEIVER_NOT_EXPORTED
        )
    }

    override fun onPause() {
        super.onPause()
        try { unregisterReceiver(sttReceiver) } catch (_: Exception) {}
    }

    private fun loadTranscript() {
        val f = File(getExternalFilesDir(null) ?: filesDir, "stt_transcript.jsonl")
        if (!f.exists()) { tvLog.text = ""; return }

        val sb = StringBuilder()
        f.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.forEach { ln ->
                val obj = runCatching { JSONObject(ln) }.getOrNull() ?: return@forEach
                val t = obj.optLong("ts", 0L)
                val text = obj.optString("text", "")
                val type = obj.optString("type", "final")

                if (text.isBlank()) return@forEach
                if (type != "final") return@forEach   // ← 跳過 partial

                val tsStr = if (t > 0) timeFmt.format(Date(t)) else "--:--:--"
                sb.append("[$tsStr] ").append(text).append('\n')  // ← 不加 [最終]
            }
        }
        tvLog.text = sb.toString()
        scrollToBottom()
    }

    private fun appendLine(line: String) {
        tvLog.append(line + "\n")
        scrollToBottom()
    }

    private fun scrollToBottom() {
        val layout = tvLog.layout ?: return
        val scrollAmount = layout.getLineTop(tvLog.lineCount) - tvLog.height
        tvLog.scrollTo(0, if (scrollAmount > 0) scrollAmount else 0)
    }
}
