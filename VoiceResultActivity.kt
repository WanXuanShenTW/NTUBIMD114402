package com.example.myapplication

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VoiceResultActivity : AppCompatActivity() {

    // === 新增：聊天資料與 UI ===
    private lateinit var rvChat: RecyclerView
    private lateinit var tvTyping: TextView
    private val items = mutableListOf<ChatMessage>()
    private lateinit var chatAdapter: ChatAdapter

    private val timeFmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    // 只顯示 partial 在 tvTyping，final 交給 ACTION_USER_UTTER
    private val sttReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != MainActivity.ACTION_STT_UPDATE) return
            val text = intent.getStringExtra(MainActivity.EXTRA_STT_TEXT).orEmpty()
            val isPartial = intent.getBooleanExtra(MainActivity.EXTRA_STT_IS_PARTIAL, false)
            if (text.isBlank() || !isPartial) {
                // 若空白或不是 partial，就隱藏「聽寫中」
                tvTyping.text = ""
                tvTyping.visibility = TextView.GONE
                return
            }
            tvTyping.text = "［長輩·聽寫中］$text…"
            tvTyping.visibility = TextView.VISIBLE
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

            val ts = System.currentTimeMillis()
            val sessionId = run {
                val sidInt = intent.getIntExtra(MainActivity.EXTRA_SESSION_ID, Int.MIN_VALUE)
                if (sidInt != Int.MIN_VALUE) sidInt.toString()
                else intent.getStringExtra(MainActivity.EXTRA_SESSION_ID).orEmpty()
            }

            when (action) {
                // ✅ 長輩 final：讀 STT 用的 key，不要再用 EXTRA_AI_TEXT
                MainActivity.ACTION_USER_UTTER -> {
                    val text =
                        intent.getStringExtra(MainActivity.EXTRA_STT_TEXT)
                            ?: intent.getStringExtra(MainActivity.EXTRA_AI_TEXT)
                            ?: ""

                    // 收到 final 時，隱藏「聽寫中」
                    tvTyping.text = ""
                    tvTyping.visibility = TextView.GONE

                    if (text.isNotBlank()) {
                        addMessage(ChatMessage(ts, Sender.ELDER, text, sessionId))
                        writeTranscript("final", text)
                    }
                }

                MainActivity.ACTION_AI_REPLY -> {
                    val text = intent.getStringExtra(MainActivity.EXTRA_AI_TEXT).orEmpty()
                    // 即使只有播語音也記一條（文字空就顯示「（AI 曾播放語音回覆）」）
                    val displayText = text.ifBlank { "（AI 曾播放語音回覆）" }
                    addMessage(ChatMessage(ts, Sender.AI, displayText, sessionId))
                    writeTranscript("ai", text.ifBlank { "" })
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_result)

        rvChat = findViewById(R.id.rvChat)
        tvTyping = findViewById(R.id.tvTyping)

        rvChat.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        chatAdapter = ChatAdapter(items, timeFmt)
        rvChat.adapter = chatAdapter

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
            // 可以視需要 Log.e
        }
    }

    private fun clearTranscript() {
        // 清空畫面資料
        items.clear()
        chatAdapter.notifyDataSetChanged()
        // 清空檔案
        val f = getTranscriptFile()
        if (f.exists()) f.delete()
        // 也把「聽寫中」收起來
        tvTyping.text = ""
        tvTyping.visibility = TextView.GONE
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

    // 載入歷史：把 "final"=長輩、"ai"=AI 轉成泡泡列表
    private fun loadTranscript() {
        val f = getTranscriptFile()
        if (!f.exists()) {
            items.clear()
            chatAdapter.notifyDataSetChanged()
            return
        }

        val list = mutableListOf<ChatMessage>()
        f.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.forEach { ln ->
                val obj = runCatching { JSONObject(ln) }.getOrNull() ?: return@forEach
                val t    = obj.optLong("ts", 0L)
                val text = obj.optString("text", "")
                val type = obj.optString("type", "final")
                val sender = if (type == "ai") Sender.AI else Sender.ELDER
                // 空字串的 AI 代表當時只有播語音，仍顯示一則提示
                val display = if (sender == Sender.AI && text.isBlank()) "（AI 曾播放語音回覆）" else text
                if (display.isNotBlank() || sender == Sender.AI) {
                    list.add(ChatMessage(t, sender, display))
                }
            }
        }
        items.clear()
        items.addAll(list)
        chatAdapter.notifyDataSetChanged()
        scrollToBottom()
    }

    private fun addMessage(msg: ChatMessage) {
        items.add(msg)
        chatAdapter.notifyItemInserted(items.lastIndex)
        scrollToBottom()
    }

    private fun scrollToBottom() {
        if (items.isNotEmpty()) {
            rvChat.scrollToPosition(items.lastIndex)
        }
    }

    // ====== 下方是簡易資料類與 Adapter（直接放同檔案最省事） ======
    enum class Sender { ELDER, AI }

    data class ChatMessage(
        val ts: Long,
        val sender: Sender,
        val text: String,
        val sessionId: String = ""
    )

    private class ChatAdapter(
        private val data: List<ChatMessage>,
        private val timeFmt: SimpleDateFormat
    ) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

        companion object {
            private const val TYPE_ELDER = 0
            private const val TYPE_AI = 1
        }

        override fun getItemViewType(position: Int): Int {
            return if (data[position].sender == Sender.ELDER) TYPE_ELDER else TYPE_AI
        }

        override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): RecyclerView.ViewHolder {
            val inflater = android.view.LayoutInflater.from(parent.context)
            return if (viewType == TYPE_ELDER) {
                val v = inflater.inflate(R.layout.item_chat_elder, parent, false)
                ElderVH(v)
            } else {
                val v = inflater.inflate(R.layout.item_chat_ai, parent, false)
                AIVH(v)
            }
        }

        override fun getItemCount(): Int = data.size

        override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
            val item = data[position]
            val tsStr = if (item.ts > 0) timeFmt.format(Date(item.ts)) else "--:--:--"
            when (holder) {
                is ElderVH -> holder.bind(item.text, tsStr, item.sessionId)
                is AIVH -> holder.bind(item.text, tsStr, item.sessionId)
            }
        }

        private class ElderVH(v: android.view.View) : RecyclerView.ViewHolder(v) {
            private val tvMsg: TextView = v.findViewById(R.id.tvMsg)
            private val tvTime: TextView = v.findViewById(R.id.tvTime)
            fun bind(text: String, time: String, sessionId: String) {
                tvMsg.text = text
                tvTime.text = if (sessionId.isNotBlank()) "$time · sess:$sessionId" else time
            }
        }

        private class AIVH(v: android.view.View) : RecyclerView.ViewHolder(v) {
            private val tvMsg: TextView = v.findViewById(R.id.tvMsg)
            private val tvTime: TextView = v.findViewById(R.id.tvTime)
            fun bind(text: String, time: String, sessionId: String) {
                tvMsg.text = text
                tvTime.text = if (sessionId.isNotBlank()) "$time · sess:$sessionId" else time
            }
        }
    }
}
