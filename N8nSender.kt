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
    private const val N8N_URL = "https://9818bd0cc929.ngrok-free.app/webhook/elder"

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
        var elderId = sp.getInt("elder_id", 1)
        if (elderId <= 0) elderId = 1
        sp.edit().putInt("elder_id", elderId).apply()

        Log.d("N8N", "POST $N8N_URL | elder_id=$elderId, session_id=$sessionId, text=$text")

        val payload = JSONObject().apply {
            put("elder_id", elderId)
            put("session_id", sessionId)
            put("text", text)
        }.toString()

        val req = Request.Builder()
            .url(N8N_URL)
            .header("Accept", "audio/*, application/json")
            .post(payload.toRequestBody(JSON))
            .build()

        client.newCall(req).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                Log.e("N8N", "request failed: ${e.message}")
                activity.runOnUiThread {
                    Toast.makeText(activity, "語音回覆失敗：${e.message}", Toast.LENGTH_SHORT).show()
                    activity.onAiSpeakingDone()
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

                    val bytes = body.bytes()
                    val ct = (body.contentType()?.toString()
                        ?: r.header("Content-Type").orEmpty()).lowercase(Locale.ROOT)

                    r.headers.forEach { (n, v) -> Log.d("N8N", "hdr $n: ${v.take(200)}") }
                    Log.d("N8N", "response code=${r.code}, ct=$ct, len=${bytes.size}")

                    val isAudio = isAudioBytes(ct, bytes)

                    if (isAudio) {
                        val ext = when {
                            ct.contains("mpeg") || ct.contains("mp3") || looksMp3(bytes) -> ".mp3"
                            ct.contains("wav") -> ".wav"
                            ct.contains("ogg") -> ".ogg"
                            else -> ".bin"
                        }
                        val dir = activity.getExternalFilesDir(null) ?: activity.filesDir
                        val outFile = File(dir, "tts_${System.currentTimeMillis()}$ext")
                        runCatching { outFile.outputStream().use { it.write(bytes) } }
                            .onFailure { Log.w("N8N", "save tts file failed: ${it.message}") }
                        Log.d("N8N", "local tts path=${outFile.absolutePath}")

                        val aiText = bestTextFromHeaders(r)
                        Log.d("N8N", "aiText(from header)='${aiText.take(120)}'")

                        activity.runOnUiThread {
                            activity.playMp3Bytes(
                                bytes,
                                onError = { e ->
                                    Log.e("N8N", "bytes playback failed: ${e.message}; try file path")
                                    activity.playTtsFromUrl(outFile.absolutePath, aiText)
                                }
                            )
                        }

                        val elderId2 = activity.getSharedPreferences("app", Context.MODE_PRIVATE).getInt("elder_id", -1)
                        val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                            setPackage(activity.packageName)
                            putExtra(MainActivity.EXTRA_ELDER_ID, elderId2)
                            putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                            putExtra(MainActivity.EXTRA_AI_TEXT, aiText)
                            putExtra(MainActivity.EXTRA_AI_AUDIO_URL, "")
                        }
                        activity.sendBroadcast(intent)

                        if (aiText.isBlank()) requestTextFallback(activity, elderId2, sessionId, text)
                        return
                    }

                    // ===== 非音訊：以 JSON 處理（支援陣列或物件；支援 audio_data.data base64） =====
                    val raw = String(bytes, Charsets.UTF_8)
                    if (raw.isBlank()) {
                        Log.w("N8N", "empty JSON")
                        activity.runOnUiThread { activity.onAiSpeakingDone() }
                        return
                    }

                    try {
                        val trimmed = raw.trim()

                        // 1) root 可能是陣列或物件
                        val rootObj = if (trimmed.startsWith("[")) {
                            val arr = org.json.JSONArray(trimmed)
                            if (arr.length() == 0) throw IllegalArgumentException("empty array JSON")
                            arr.getJSONObject(0)
                        } else {
                            JSONObject(trimmed)
                        }

                        // 2) 優先：audio_data.data (base64 MP3)
                        if (rootObj.has("audio_data")) {
                            val audioObj = rootObj.getJSONObject("audio_data")
                            val b64 = audioObj.optString("data", "").replace("\\s".toRegex(), "")
                            if (b64.isNotBlank()) {
                                val mp3Bytes = try {
                                    android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                                } catch (_: IllegalArgumentException) {
                                    android.util.Base64.decode(b64, android.util.Base64.URL_SAFE)
                                }

                                activity.runOnUiThread {
                                    activity.playMp3Bytes(
                                        bytes = mp3Bytes,
                                        onError = { e ->
                                            Log.e("N8N", "play base64 bytes failed: ${e.message}")
                                            activity.onAiSpeakingDone()
                                        }
                                    )

                                    val answer = bestTextFromJson(rootObj)
                                    val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                                        setPackage(activity.packageName)
                                        putExtra(MainActivity.EXTRA_ELDER_ID, elderId)
                                        putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                                        putExtra(MainActivity.EXTRA_AI_TEXT, answer)
                                        putExtra(MainActivity.EXTRA_AI_AUDIO_URL, "")
                                    }
                                    activity.sendBroadcast(intent)
                                }
                                return
                            }
                        }

                        // 3) 相容：audio_base64（字串）
                        val j = rootObj
                        val b64Raw = j.optString("audio_base64", "")
                        val b64 = b64Raw.replace("\\s".toRegex(), "")
                        if (b64.isNotEmpty()) {
                            val mp3Bytes = try {
                                android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                            } catch (_: IllegalArgumentException) {
                                android.util.Base64.decode(b64, android.util.Base64.URL_SAFE)
                            }

                            activity.runOnUiThread {
                                activity.playMp3Bytes(mp3Bytes)
                                val answer = bestTextFromJson(j)
                                val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                                    setPackage(activity.packageName)
                                    putExtra(MainActivity.EXTRA_ELDER_ID, elderId)
                                    putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                                    putExtra(MainActivity.EXTRA_AI_TEXT, answer)
                                    putExtra(MainActivity.EXTRA_AI_AUDIO_URL, "")
                                }
                                activity.sendBroadcast(intent)
                            }
                            return
                        }

                        // 4) 最後：audio_url / 文字
                        val answer = bestTextFromJson(j)
                        val url = j.optString("audio_url", "")

                        val intent = Intent(MainActivity.ACTION_AI_REPLY).apply {
                            setPackage(activity.packageName)
                            putExtra(MainActivity.EXTRA_ELDER_ID, elderId)
                            putExtra(MainActivity.EXTRA_SESSION_ID, sessionId)
                            putExtra(MainActivity.EXTRA_AI_TEXT, answer)
                            putExtra(MainActivity.EXTRA_AI_AUDIO_URL, url)
                        }
                        activity.sendBroadcast(intent)

                    } catch (e: Exception) {
                        Log.e("N8N", "parse response error: ${e.message} ; raw='${raw.take(200)}'")
                        activity.runOnUiThread {
                            Toast.makeText(activity, "回應格式錯誤", Toast.LENGTH_SHORT).show()
                            activity.onAiSpeakingDone()
                        }
                    }
                }
            }
        })
    }

    private fun isAudioBytes(ct: String, bytes: ByteArray): Boolean {
        // 用檔頭特徵先判斷
        if (looksMp3(bytes)) return true // "ID3"
        if (bytes.size >= 12 &&
            bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() &&
            bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte()
        ) return true // WAV
        if (bytes.size >= 4 &&
            bytes[0] == 'O'.code.toByte() && bytes[1] == 'g'.code.toByte() &&
            bytes[2] == 'g'.code.toByte() && bytes[3] == 'S'.code.toByte()
        ) return true // OGG
        // MP3 frame sync (0xFFE*)
        if (bytes.size >= 2 && (bytes[0].toInt() and 0xFF) == 0xFF && ((bytes[1].toInt() and 0xE0) == 0xE0))
            return true

        // 只靠 header 宣稱 audio，不可信；限制至少 1KB
        val ctLower = ct.lowercase(Locale.ROOT)
        if (ctLower.startsWith("audio/") && bytes.size >= 1024) return true

        return false
    }


    private fun looksMp3(bytes: ByteArray): Boolean =
        bytes.size >= 3 && bytes[0] == 0x49.toByte() && bytes[1] == 0x44.toByte() && bytes[2] == 0x33.toByte() // "ID3"

    private fun bestTextFromHeaders(resp: okhttp3.Response): String {
        fun clean(v: String?): String {
            val s = v?.trim().orEmpty()
            val low = s.lowercase(Locale.ROOT)
            return if (s.isEmpty() || low == "undefined" || low == "null") "" else s
        }

        // 先找 URL-encoded 的文字（支援 ai-text / x-ai-text / user-text / x-text）
        val urlEncKeys = listOf(
            "X-AI-Text", "x-ai-text", "AI-Text", "ai-text",
            "X-User-Text", "x-user-text", "User-Text", "user-text",
            "X-Text", "x-text", "Text", "text"
        )
        for (k in urlEncKeys) {
            val raw = clean(resp.header(k))
            if (raw.isNotBlank()) {
                return try {
                    // 伺服器用 encodeURIComponent -> 這裡用 URLDecoder.decode 還原
                    java.net.URLDecoder.decode(raw, "UTF-8")
                } catch (_: Exception) {
                    raw // 解不動就原樣返回
                }
            }
        }

        // 再找 Base64（支援 ai-text-base64 / x-ai-text-base64 / ...）
        val b64Keys = listOf(
            "X-AI-Text-Base64", "x-ai-text-base64", "AI-Text-Base64", "ai-text-base64",
            "X-User-Text-Base64", "x-user-text-base64", "User-Text-Base64", "user-text-base64",
            "X-Text-Base64", "x-text-base64", "Text-Base64", "text-base64"
        )
        for (k in b64Keys) {
            val b64 = clean(resp.header(k))
            if (b64.isNotBlank()) {
                return try {
                    val bytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                    String(bytes, Charsets.UTF_8)
                } catch (_: IllegalArgumentException) {
                    try {
                        val bytes = android.util.Base64.decode(b64, android.util.Base64.URL_SAFE)
                        String(bytes, Charsets.UTF_8)
                    } catch (_: Exception) {
                        ""
                    }
                }
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
