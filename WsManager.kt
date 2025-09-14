package com.example.myapplication

import android.util.Log
import okhttp3.*

class WsManager(private val url: String) {
    private val client = OkHttpClient.Builder().retryOnConnectionFailure(true).build()
    private var socket: WebSocket? = null
    @Volatile var isConnected: Boolean = false
        private set

    fun connect(
        onState: ((Boolean, String?) -> Unit)? = null,
        onMessage: ((String) -> Unit)? = null
    ) {
        if (socket != null) return
        val req = Request.Builder().url(url).build()
        socket = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isConnected = true
                onState?.invoke(true, null)
                Log.i("WS", "connected: $url")
            }
            override fun onMessage(ws: WebSocket, text: String) {
                Log.d("WS", "recv: $text")
                onMessage?.invoke(text)   // ← 把文字回傳給 Activity
            }
            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                onState?.invoke(false, t.message)
                Log.e("WS", "failure: ${t.message}", t)
                socket = null
            }
            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                isConnected = false
                onState?.invoke(false, "closed: $code/$reason")
                socket = null
            }
        })
    }

    fun send(text: String): Boolean = socket?.send(text) ?: false

    fun close(code: Int = 1000, reason: String = "bye") {
        runCatching { socket?.close(code, reason) }
        socket = null
        isConnected = false
    }
}
