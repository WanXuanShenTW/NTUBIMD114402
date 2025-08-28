package com.example.myapplication

import android.content.Context
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

object N8nSender {

    // 你的 n8n Webhook
    private const val N8N_URL =
        "https://7a37116c8e0a.ngrok-free.app/webhook/elder-voice"

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(12, TimeUnit.SECONDS)
            .writeTimeout(12, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }

    private val JSON = "application/json; charset=utf-8".toMediaType()

    fun sendElderVoice(context: Context, text: String) {
        if (text.isBlank()) return

        val sp = context.getSharedPreferences("app", Context.MODE_PRIVATE)
        val userId = sp.getInt("user_id", -1)
        val contactId = sp.getString("contact_id", null) ?: ""

        if (userId <= 0 || contactId.isBlank()) {
            Log.e("N8N", "user_id 或 contact_id 不完整，略過送出")
            return
        }

        val payload = JSONObject().apply {
            put("user_id", userId)
            put("contact_id", contactId)
            put("text", text)
        }.toString()

        Log.d("N8N", "POST $N8N_URL payload=$payload")

        val body: RequestBody = payload.toRequestBody(JSON)

        val req = Request.Builder()
            .url(N8N_URL)
            .post(body)
            .build()

        client.newCall(req).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                Log.e("N8N", "送出失敗: ${e.message}")
            }
            override fun onResponse(call: okhttp3.Call, resp: okhttp3.Response) {
                val bodyStr = resp.body?.string().orEmpty()
                Log.d("N8N", "送出成功: code=${resp.code}, body=$bodyStr")
                resp.close()
            }
        })
    }
}
