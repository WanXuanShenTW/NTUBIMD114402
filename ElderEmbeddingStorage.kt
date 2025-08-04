package com.example.myapplication

import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ElderEmbeddingStorage {
    private const val TAG = "ElderEmbeddingStorage"
    private const val FILE_NAME = "elder_embedding.vec"

    private fun file(context: Context): File =
        File(context.filesDir, FILE_NAME) // → 一律用內部儲存，避免權限/路徑不一致

    fun save(context: Context, embedding: FloatArray) {
        // 原子寫入：先寫 tmp，再 rename
        val tmp = File(context.filesDir, "$FILE_NAME.tmp")
        val outBuf = ByteBuffer
            .allocate(4 + embedding.size * 4) // 4 bytes 長度 + N*4 bytes float
            .order(ByteOrder.LITTLE_ENDIAN)
        outBuf.putInt(embedding.size)
        for (v in embedding) outBuf.putFloat(v)
        tmp.outputStream().use { it.write(outBuf.array()) }
        val target = file(context)
        if (target.exists()) target.delete()
        val ok = tmp.renameTo(target)
        Log.d(TAG, "save -> path=${target.absolutePath}, size=${embedding.size}, ok=$ok")
    }

    fun load(context: Context): FloatArray? {
        val f = file(context)
        if (!f.exists() || f.length() < 4) {
            Log.d(TAG, "load -> file not found or too small")
            return null
        }
        val bytes = f.readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val n = buf.int
        if (n <= 0 || bytes.size != 4 + n * 4) {
            Log.w(TAG, "load -> invalid length n=$n, bytes=${bytes.size}")
            return null
        }
        val arr = FloatArray(n)
        for (i in 0 until n) arr[i] = buf.float
        Log.d(TAG, "load -> path=${f.absolutePath}, size=$n")
        return arr
    }

    fun clear(context: Context): Boolean {
        val f = file(context)
        val ok = if (f.exists()) f.delete() else true
        Log.d(TAG, "clear -> ${f.absolutePath}, ok=$ok")
        return ok
    }

    fun exists(context: Context): Boolean = file(context).exists()
}
