package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.util.Log
import android.widget.Toast
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.TimeUnit

object N8nSender {

    // n8n webhook
    private const val N8N_URL = "https://a7a35ec77cab.ngrok-free.app/webhook/elder"

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .callTimeout(180, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }
    private val JSON = "application/json; charset=utf-8".toMediaType()

    // 接受 Int 型別的 sessionId
    fun sendElderVoiceAndSpeak(activity: MainActivity, text: String, sessionId: Int) {
        if (text.isBlank()) return

        val sp = activity.getSharedPreferences("app", Context.MODE_PRIVATE)
        var elderId = sp.getInt("elder_id", 1)   // 測試期預設 1
        if (elderId <= 0) elderId = 1
        sp.edit().putInt("elder_id", elderId).apply()

        Log.d("N8N", "POST $N8N_URL | elder_id=$elderId, session_id=$sessionId, text=$text")

        // JSON payload（session_id 為 JSON number）
        val payload = JSONObject().apply {
            put("elder_id", elderId)
            put("session_id", sessionId)
            put("text", text)
        }.toString()

        val req = Request.Builder()
            .url(N8N_URL)
            .addHeader("Accept", "application/json, audio/*") // ← 優先 JSON，但也允許 audio
            .post(payload.toRequestBody(JSON))
            .build()

        client.newCall(req).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                Log.e("N8N", "request failed: ${e.message}")
                activity.runOnUiThread {
                    Toast.makeText(activity, "語音回覆失敗：${e.message}", Toast.LENGTH_SHORT).show()
                    activity.onAiSpeakingDone() // 確保循環恢復
                }
            }

            override fun onResponse(call: okhttp3.Call, resp: okhttp3.Response) {
                resp.use { r ->
                    val body = r.body
                    if (!r.isSuccessful || body == null) {
                        val err = runCatching { r.body?.string().orEmpty() }.getOrDefault("")
                        Log.e("N8N", "http not successful: ${r.code}, body='${err.take(200)}'")
                        activity.runOnUiThread { activity.onAiSpeakingDone() }
                        return
                    }

                    // 讀 bytes 後再判斷 audio / json，避免 content-type 不準
                    val bytes = body.bytes()
                    val ct = (body.contentType()?.toString()
                        ?: r.header("Content-Type").orEmpty()).lowercase(Locale.ROOT)

                    // 除錯：列印回應 header（一次就能看出伺服器到底回了什麼）
                    r.headers.forEach { (n, v) -> Log.d("N8N", "hdr $n: ${v.take(200)}") }
                    Log.d("N8N", "response code=${r.code}, ct=$ct, len=${bytes.size}")

                    val isAudio = isAudioBytes(ct, bytes)

                    if (isAudio) {
                        // 儲存音檔
                        val ext = when {
                            ct.contains("mpeg") || ct.contains("mp3") || looksMp3(bytes) -> ".mp3"
                            ct.contains("wav") -> ".wav"
                            ct.contains("ogg") -> ".ogg"
                            else -> ".bin"
                        }
                        val dir = activity.getExternalFilesDir(null) ?: activity.filesDir
                        val outFile = File(dir, "tts_${System.currentTimeMillis()}$ext")
                        outFile.outputStream().use { it.write(bytes) }

                        // 也嘗試從 header 抓文字（若你在 n8n 回了 X-AI-Text / X-AI-Text-Base64）
                        val aiText = bestTextFromHeaders(r)
                        Log.d("N8N", "aiText(from header)='${aiText.take(120)}'")

                        // 廣播：MainActivity 會負責播放與 UI；VoiceResult 會顯示
                        val elderId2 = activity.getSharedPreferences("app", Context.MODE_PRIVATE).getInt("elder_id", -1)
                        val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                            setPackage(activity.packageName)
                            putExtra(MainActivity.EXTRA_ELDER_ID, elderId2)
                            putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)   // Int
                            putExtra(MainActivity.EXTRA_AI_TEXT, aiText)                    // 可能為空
                            putExtra(MainActivity.EXTRA_AI_AUDIO_URL, outFile.absolutePath)  // 本地音檔
                        }
                        activity.sendBroadcast(intent)

                        // 若 header 沒文字、但你需要顯示文字，可取消註解，啟用「後補 JSON 取文字」
                        // if (aiText.isBlank()) requestTextFallback(activity, elderId2, sessionId, text)

                        return
                    }

                    // 非音訊 → 當 JSON 處理
                    val raw = String(bytes, Charsets.UTF_8)
                    if (raw.isBlank()) {
                        Log.w("N8N", "empty JSON")
                        activity.runOnUiThread { activity.onAiSpeakingDone() }
                        return
                    }

                    val j = try {
                        JSONObject(raw)
                    } catch (e: Exception) {
                        Log.e("N8N", "parse response error: ${e.message} ; raw='${raw.take(200)}'")
                        activity.runOnUiThread {
                            Toast.makeText(activity, "回應格式錯誤", Toast.LENGTH_SHORT).show()
                            activity.onAiSpeakingDone()
                        }
                        return
                    }

                    val answer = bestTextFromJson(j) // 兼容 answer_text/text/message/reply…
                    val url = j.optString("audio_url", "")

                    val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                        setPackage(activity.packageName)
                        putExtra(MainActivity.EXTRA_ELDER_ID, elderId)
                        putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                        putExtra(MainActivity.EXTRA_AI_TEXT, answer)
                        putExtra(MainActivity.EXTRA_AI_AUDIO_URL, url)
                    }
                    activity.sendBroadcast(intent)
                }
            }
        })
    }

    // ——— helpers ———

    private fun isAudioBytes(ct: String, bytes: ByteArray): Boolean {
        if (ct.startsWith("audio/") || ct == "application/octet-stream") return true
        // "ID3" 標頭（MP3）
        if (looksMp3(bytes)) return true
        // MPEG frame sync
        if (bytes.size >= 2 && (bytes[0].toInt() and 0xFF) == 0xFF && ((bytes[1].toInt() and 0xE0) == 0xE0)) return true
        return false
    }

    private fun looksMp3(bytes: ByteArray): Boolean =
        bytes.size >= 3 && bytes[0] == 0x49.toByte() && bytes[1] == 0x44.toByte() && bytes[2] == 0x33.toByte() // "ID3"

    private fun bestTextFromHeaders(resp: okhttp3.Response): String {
        fun header(vararg keys: String): String {
            for (k in keys) {
                val v = resp.header(k)
                if (!v.isNullOrBlank()) return v
            }
            return ""
        }
        // 先試 URL encoded
        runCatching {
            val h = header("X-AI-Text", "x-ai-text", "X-Text", "x-text")
            if (h.isNotBlank()) return java.net.URLDecoder.decode(h, "UTF-8")
        }
        // 再試 Base64
        runCatching {
            val b64 = header("X-AI-Text-Base64", "x-ai-text-base64", "X-Text-Base64", "x-text-base64")
            if (b64.isNotBlank()) {
                val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                return String(bytes, Charsets.UTF_8)
            }
        }
        return ""
    }

    private fun bestTextFromJson(j: JSONObject): String {
        val keys = listOf("answer_text", "text", "message", "reply", "output")
        for (k in keys) {
            val v = j.optString(k, "")
            if (v.isNotBlank()) return v
        }
        return ""
    }

    // （選用）後補打 JSON 只拿文字；需要時把呼叫處的註解打開即可
    private fun requestTextFallback(activity: MainActivity, elderId: Int, sessionId: Int, userText: String) {
        try {
            val payload = JSONObject().apply {
                put("elder_id", elderId)
                put("session_id", sessionId)
                put("text", userText)
            }.toString()

            val req = Request.Builder()
                .url(N8N_URL)
                .addHeader("Accept", "application/json")
                .addHeader("X-Prefer-JSON", "1")
                .post(payload.toRequestBody(JSON))
                .build()

            client.newCall(req).enqueue(object : okhttp3.Callback {
                override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                    Log.w("N8N", "fallback JSON failed: ${e.message}")
                }

                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    response.use { r ->
                        val raw = r.body?.string().orEmpty()
                        if (!r.isSuccessful || raw.isBlank()) return
                        val j = runCatching { JSONObject(raw) }.getOrNull() ?: return
                        val aiText = bestTextFromJson(j)
                        if (aiText.isBlank()) return

                        val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                            setPackage(activity.packageName)
                            putExtra(MainActivity.EXTRA_ELDER_ID, elderId)
                            putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                            putExtra(MainActivity.EXTRA_AI_TEXT, aiText)
                            putExtra(MainActivity.EXTRA_AI_AUDIO_URL, "")
                        }
                        activity.sendBroadcast(intent)
                    }
                }
            })
        } catch (e: Exception) {
            Log.e("N8N", "fallback JSON error", e)
        }
    }
}
