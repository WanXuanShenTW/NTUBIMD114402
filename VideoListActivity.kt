package com.example.myapplication

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.*
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
    private val adapter by lazy { FallRecordAdapter(::onItemClick) }

    private lateinit var spinnerCategory: Spinner
    private enum class Category { FALL, INTERACTION }
    private var currentCategory: Category = Category.FALL

    // 只在第一次進來時自動打開選單
    private var openedOnce = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_list)

        swipe = findViewById(R.id.swipe)
        recycler = findViewById(R.id.recyclerView)
        progress = findViewById(R.id.progressBar)
        emptyState = findViewById(R.id.emptyState)
        empty = findViewById(R.id.emptyView)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        spinnerCategory = findViewById(R.id.spinnerCategory)

        // 第 0 項是提示（灰色、不可選）
        val categories = listOf("選擇類別", "跌倒列表", "互動報告")

        val spinAdapter = object : ArrayAdapter<String>(
            this, R.layout.item_spinner_text, categories
        ) {
            override fun isEnabled(position: Int): Boolean = position != 0
            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val v = super.getDropDownView(position, convertView, parent) as TextView
                v.setTextColor(if (position == 0) 0xFF9E9E9E.toInt() else 0xFF333333.toInt())
                return v
            }
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val v = super.getView(position, convertView, parent) as TextView
                v.setTextColor(if (position == 0) 0xFF9E9E9E.toInt() else 0xFF333333.toInt())
                return v
            }
        }.also { it.setDropDownViewResource(R.layout.item_spinner_text) }

        spinnerCategory.adapter = spinAdapter
        spinnerCategory.setSelection(0, false) // 停在「選擇類別」，不觸發 onItemSelected

        spinnerCategory.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                when (position) {
                    0 -> {
                        // 初始或切回提示
                        showEmpty("請先選擇類別")
                        swipe.isRefreshing = false
                    }
                    1 -> { // 跌倒列表
                        currentCategory = Category.FALL
                        loadData(isRefresh = true)
                    }
                    2 -> { // 互動報告（先顯示占位文字，之後接 API）
                        currentCategory = Category.INTERACTION
                        recycler.visibility = View.GONE
                        progress.visibility = View.GONE
                        emptyState.visibility = View.VISIBLE
                        empty.text = "互動報告功能開發中"
                        swipe.isRefreshing = false
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter = adapter

        // 下拉刷新：依目前類別重載
        swipe.setOnRefreshListener {
            when (spinnerCategory.selectedItemPosition) {
                1 -> loadData(isRefresh = true)
                2 -> swipe.isRefreshing = false // 互動報告未接 API，先收起
                else -> swipe.isRefreshing = false
            }
        }

        // 頁面初始顯示提示
        showEmpty("請先選擇類別")
    }

    override fun onResume() {
        super.onResume()
        // 第一次進頁就自動打開 Spinner
        if (!openedOnce && spinnerCategory.selectedItemPosition == 0) {
            openedOnce = true
            spinnerCategory.post { spinnerCategory.performClick() }
        }
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

    private val elderChangedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (currentCategory == Category.FALL) {
                loadData(isRefresh = true)
            }
        }
    }

    override fun onStop() {
        runCatching { unregisterReceiver(elderChangedReceiver) }
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
                    // 新→舊
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
        val detail = """
            時間：${formatTime(item.detected_time)}
            地點：${item.location ?: "未知"}
            跌倒前：${item.pose_before_fall ?: "—"}
        """.trimIndent()
        androidXAlert(detail)
    }

    private fun parseEpoch(raw: String): Long {
        for (p in listOf(
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        )) {
            try {
                val sdf = SimpleDateFormat(p, Locale.getDefault())
                // 若後端是 UTC 就打開下一行
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

    private fun androidXAlert(message: String) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("事件詳情")
            .setMessage(message)
            .setPositiveButton("關閉", null)
            .show()
    }
}
