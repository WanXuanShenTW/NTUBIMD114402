package com.example.myapplication

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VoiceResultActivity : AppCompatActivity() {

    private lateinit var tvLog: TextView
    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // 只顯示 partial，final 交給 ACTION_USER_UTTER（避免重複）
    private val sttReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != MainActivity.ACTION_STT_UPDATE) return
            val text = intent.getStringExtra(MainActivity.EXTRA_STT_TEXT).orEmpty()
            val isPartial = intent.getBooleanExtra(MainActivity.EXTRA_STT_IS_PARTIAL, false)
            if (text.isBlank() || !isPartial) return

            val ts = timeFmt.format(Date())
            appendLine("[$ts] [長輩·聽寫中] $text…")
        }
    }

    private fun getTranscriptFile(): File {
        val sp = getSharedPreferences("app", Context.MODE_PRIVATE)
        val elderId = sp.getInt("elder_id", -1)
        val fname = if (elderId > 0) {
            "stt_transcript_elder${elderId}.jsonl"
        } else {
            "stt_transcript.jsonl"
        }
        return File(getExternalFilesDir(null) ?: filesDir, fname)
    }

    private val chatReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            if (intent == null) return
            val action = intent.action ?: return

            val elderIdFromIntent = intent.getIntExtra(MainActivity.EXTRA_ELDER_ID, -1)
            val currentElderId = getSharedPreferences("app", Context.MODE_PRIVATE)
                .getInt("elder_id", 1)

            if (elderIdFromIntent > 0 && elderIdFromIntent != currentElderId) {
                return
            }
            val ts = timeFmt.format(Date())
            val sessionId = run {
                val sidInt = intent.getIntExtra(MainActivity.EXTRA_SESSION_ID, Int.MIN_VALUE)
                if (sidInt != Int.MIN_VALUE) sidInt.toString()
                else intent.getStringExtra(MainActivity.EXTRA_SESSION_ID).orEmpty()
            }

            when (action) {
                // ✅ 長輩 final：讀 STT 用的 key，不要再用 EXTRA_AI_TEXT
                MainActivity.ACTION_USER_UTTER -> {
                    val text =
                        intent.getStringExtra(MainActivity.EXTRA_STT_TEXT) // STT final 用的 key
                            ?: intent.getStringExtra(MainActivity.EXTRA_AI_TEXT) // 最後容錯
                            ?: ""

                    if (text.isNotBlank()) {
                        appendLine("[$ts] [長輩] $text" + sessionSuffix(sessionId))
                        writeTranscript("final", text)
                    }
                }

                MainActivity.ACTION_AI_REPLY -> {
                    val text = intent.getStringExtra(MainActivity.EXTRA_AI_TEXT).orEmpty()
                    val audioUrl = intent.getStringExtra(MainActivity.EXTRA_AI_AUDIO_URL).orEmpty()
                    val show = when {
                        text.isNotBlank() && audioUrl.isNotBlank() -> "$text"
                        text.isNotBlank() -> text
                        audioUrl.isNotBlank() -> "（AI 已回覆並播放語音）"
                        else -> "（AI 回覆為空）"
                    }
                    appendLine("[$ts] [AI] $show" + sessionSuffix(sessionId))
                    // 即使只播了語音也記錄下來，方便之後回看
                    writeTranscript("ai", text.ifBlank { "" })
                }
            }
        }
    }
    private fun sessionSuffix(sessionId: String): String =
        if (sessionId.isNotBlank()) "  (sess:$sessionId)" else ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_result)

        tvLog = findViewById(R.id.tvLog)
        tvLog.movementMethod = ScrollingMovementMethod()

        loadTranscript() // 載入歷史 final/ai 記錄

        // 返回主頁
        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        findViewById<ImageButton>(R.id.btnClearTranscript).setOnClickListener {
            clearTranscript()
            Toast.makeText(this, "已清除紀錄", Toast.LENGTH_SHORT).show()
        }
    }

    private fun writeTranscript(type: String, text: String) {
        try {
            val obj = JSONObject().apply {
                put("ts", System.currentTimeMillis())
                put("type", type) // "final" 或 "ai"
                put("text", text)
            }
            val f = getTranscriptFile()
            f.parentFile?.mkdirs()
            f.appendText(obj.toString() + "\n", Charsets.UTF_8)
        } catch (_: Exception) {
            // 可視需要加上 Log.e
        }
    }

    private fun clearTranscript() {
        tvLog.text = ""
        val f = getTranscriptFile()
        if (f.exists()) {
            f.delete()
        }
    }

    override fun onResume() {
        super.onResume()
        ContextCompat.registerReceiver(
            this,
            sttReceiver,
            IntentFilter(MainActivity.ACTION_STT_UPDATE),
            null, null,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )

        // 長輩與 AI 的對話廣播
        val chatFilter = IntentFilter().apply {
            addAction(MainActivity.ACTION_USER_UTTER)
            addAction(MainActivity.ACTION_AI_REPLY)
        }
        ContextCompat.registerReceiver(
            this,
            chatReceiver,
            chatFilter,
            null, null,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
    }

    override fun onPause() {
        super.onPause()
        runCatching { unregisterReceiver(sttReceiver) }
        runCatching { unregisterReceiver(chatReceiver) }
    }

    // 載入歷史：同時讀「final」(長輩) 與「ai」(AI)
    private fun loadTranscript() {
        val f = getTranscriptFile()
        if (!f.exists()) {
            tvLog.text = ""
            return
        }

        val sb = StringBuilder()
        f.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.forEach { ln ->
                val obj = runCatching { JSONObject(ln) }.getOrNull() ?: return@forEach
                val t    = obj.optLong("ts", 0L)
                val text = obj.optString("text", "")
                val type = obj.optString("type", "final") // "final"=長輩, "ai"=AI
                val tsStr = if (t > 0) timeFmt.format(Date(t)) else "--:--:--"

                when (type) {
                    "final" -> if (text.isNotBlank())
                        sb.append("[$tsStr] [長輩] ").append(text).append('\n')
                    "ai"    -> sb.append("[$tsStr] [AI] ")
                        .append(text.ifBlank { "（AI 曾播放語音回覆）🔊" })
                        .append('\n')
                    else    -> { /* 其他類型略過 */ }
                }
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
