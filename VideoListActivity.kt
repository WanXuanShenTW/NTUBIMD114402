package com.example.myapplication

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout
import com.example.myapplication.adapter.FallRecordAdapter
import com.example.myapplication.model.FallRecord
import com.example.myapplication.network.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

class VideoListActivity : AppCompatActivity() {

    private lateinit var swipe: SwipeRefreshLayout
    private lateinit var recycler: RecyclerView
    private lateinit var progress: ProgressBar
    private lateinit var emptyState: View
    private lateinit var empty: TextView
    private lateinit var btnRetry: Button
    private val adapter by lazy { FallRecordAdapter(::onItemClick) }

    // 切換被照護者 → 自動刷新
    private val elderChangedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            loadData()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_list)

        swipe = findViewById(R.id.swipe)
        recycler = findViewById(R.id.recyclerView)
        progress = findViewById(R.id.progressBar)
        emptyState = findViewById(R.id.emptyState)
        empty = findViewById(R.id.emptyView)
        btnRetry = findViewById(R.id.btnRetry)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter = adapter

        swipe.setOnRefreshListener { loadData(isRefresh = true) }
        btnRetry.setOnClickListener { loadData() }

        loadData()
    }

    override fun onStart() {
        super.onStart()
        val filter = IntentFilter(AppKeys.ACTION_ELDER_CHANGED)
        ContextCompat.registerReceiver(
            this,
            elderChangedReceiver,
            filter,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
    }

    override fun onStop() {
        try { unregisterReceiver(elderChangedReceiver) } catch (_: Exception) {}
        super.onStop()
    }

    private fun loadData(isRefresh: Boolean = false) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val elderId = sp.getInt(AppKeys.ELDER_ID, -1)
        if (elderId <= 0) {
            showEmpty("尚未選擇被照護者")
            return
        }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        lifecycleScope.launch {
            try {
                val resp = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getFallEventRecords(elderId)
                }
                swipe.isRefreshing = false

                val records = resp.data?.records ?: emptyList()
                if (resp.success && records.isNotEmpty()) {
                    // 排序（新→舊）
                    val sorted = records.sortedByDescending { parseEpoch(it.detected_time) }

                    fun normalizeTime(s: String) = s.replace('T', ' ').removeSuffix("Z")
                    val normalized = sorted.map { it.copy(detected_time = normalizeTime(it.detected_time)) }

                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                    adapter.submitList(normalized)
                } else {
                    showEmpty("這位被照護者目前沒有跌倒事件")
                }
            } catch (_: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
            }
        }
    }

    private fun showEmpty(msg: String) {
        recycler.visibility = View.GONE
        progress.visibility = View.GONE
        emptyState.visibility = View.VISIBLE
        empty.text = msg
    }

    private fun onItemClick(item: FallRecord) {
        // 顯示簡單詳情（無影片）
        val detail = """
            時間：${formatTime(item.detected_time)}
            地點：${item.location ?: "未知"}
            跌倒前：${item.pose_before_fall ?: "—"}
        """.trimIndent()
        androidXAlert(detail)
    }

    // ---- 時間處理 ----
    private fun parseEpoch(raw: String): Long {
        for (p in listOf(
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        )) {
            try {
                val sdf = SimpleDateFormat(p, Locale.getDefault())
                // 如果你的後端用本地時間，就用預設時區；若是 UTC，把下一行改成 UTC
                // sdf.timeZone = TimeZone.getTimeZone("UTC")
                return sdf.parse(raw)?.time ?: Long.MIN_VALUE
            } catch (_: Exception) {}
        }
        return Long.MIN_VALUE
    }

    private fun formatTime(raw: String): String {
        val epoch = parseEpoch(raw)
        if (epoch == Long.MIN_VALUE) return raw
        val out = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.getDefault())
        return out.format(Date(epoch))
    }

    // 簡單的 appcompat 對話框
    private fun androidXAlert(message: String) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("事件詳情")
            .setMessage(message)
            .setPositiveButton("關閉", null)
            .show()
    }
}
