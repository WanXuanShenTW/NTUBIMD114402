package com.example.myapplication

import android.content.Context
import kotlin.math.pow

object SessionId {
    private const val KEY = "stt_session_seq"

    private const val PREFIX = "sess"
    private const val WIDTH = 3
    private val MAX = 10.0.pow(WIDTH).toInt() - 1  // 999

    @Synchronized
    fun next(context: Context): Int {
        val sp  = context.getSharedPreferences("app", Context.MODE_PRIVATE)
        val cur = sp.getInt(KEY, 0)
        val nxt = if (cur >= MAX) 1 else cur + 1
        sp.edit().putInt(KEY, nxt).apply()
        return nxt
    }

    @Synchronized
    fun current(context: Context): Int {
        val sp = context.getSharedPreferences("app", Context.MODE_PRIVATE)
        return sp.getInt(KEY, 0)
    }

    @Synchronized
    fun reset(context: Context) {
        val sp = context.getSharedPreferences("app", Context.MODE_PRIVATE)
        sp.edit().putInt(KEY, 0).apply()
    }

    fun format(sessionId: Int): String =
        PREFIX + sessionId.toString().padStart(WIDTH, '0')
}
