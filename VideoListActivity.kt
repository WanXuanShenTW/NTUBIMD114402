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
import com.example.myapplication.model.SleepRecord
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class VideoListActivity : AppCompatActivity() {

    private lateinit var swipe: SwipeRefreshLayout
    private lateinit var recycler: RecyclerView
    private lateinit var progress: ProgressBar
    private lateinit var emptyState: View
    private lateinit var empty: TextView
    private val adapter by lazy { FallRecordAdapter(::onItemClick) }
    private enum class Category { FALL, SLEEP, INTERACTION }

    private lateinit var spinnerCategory: Spinner
    private var currentCategory: Category = Category.FALL
    private val sleepAdapter by lazy { com.example.myapplication.adapter.SleepRecordAdapter(::onSleepClick) }

    private var openedOnce = false

    private var startTimeStr: String? = null
    private var endTimeStr:   String? = null
    private var sleepDate: String = todayForSleep()

    private enum class SleepMode { DAILY, WEEKLY }
    private var sleepMode: SleepMode = SleepMode.DAILY
    private var fallRangeInitialized = false

    private lateinit var textRange: TextView
    private lateinit var barDateNav: LinearLayout
    private lateinit var btnPrevDay: ImageButton
    private lateinit var btnNextDay: ImageButton


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_list)

        swipe = findViewById(R.id.swipe)
        recycler = findViewById(R.id.recyclerView)
        progress = findViewById(R.id.progressBar)
        emptyState = findViewById(R.id.emptyState)
        empty = findViewById(R.id.emptyView)
        textRange   = findViewById(R.id.textRange)
        barDateNav  = findViewById(R.id.barDateNav)
        btnPrevDay  = findViewById(R.id.btnPrevDay)
        btnNextDay  = findViewById(R.id.btnNextDay)

        btnPrevDay.setOnClickListener {
            if (currentCategory == Category.SLEEP) {
                shiftSleepDate(-1)
                updateSleepDateLabel()
                loadSleepDaily(isRefresh = true)
            }
        }
        btnNextDay.setOnClickListener {
            if (currentCategory == Category.SLEEP) {
                shiftSleepDate(+1)
                updateSleepDateLabel()
                loadSleepDaily(isRefresh = true)
            }
        }

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        spinnerCategory = findViewById(R.id.spinnerCategory)

        val categories = listOf("— 選擇類別", "跌倒列表", "睡眠列表", "互動報告")

        val icons = listOf(0, R.drawable.ic_fall_24, R.drawable.ic_sleep_24, R.drawable.ic_report_24)

        val spinAdapter = object : ArrayAdapter<String>(
            this, R.layout.item_spinner_icon, categories
        ) {
            override fun isEnabled(position: Int) = position != 0

            private fun colorFor(position: Int) =
                if (position == 0) 0xFF9E9E9E.toInt() else 0xFF333333.toInt()

            private fun bind(tv: TextView, position: Int) {
                tv.setTextColor(colorFor(position))
                val iconRes = icons.getOrNull(position) ?: 0
                val icon = if (position == 0 || iconRes == 0) null
                else ContextCompat.getDrawable(context, iconRes)
                tv.setCompoundDrawablesRelativeWithIntrinsicBounds(icon, null, null, null)
            }

            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val tv = super.getView(position, convertView, parent) as TextView
                bind(tv, position)
                return tv
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val tv = super.getDropDownView(position, convertView, parent) as TextView
                bind(tv, position)
                return tv
            }
        }.also {
            it.setDropDownViewResource(R.layout.item_spinner_icon)
        }

        spinnerCategory.adapter = spinAdapter
        spinnerCategory.setSelection(0, false)

        findViewById<ImageButton>(R.id.btnFilter).setOnClickListener { v ->
            val popup = android.widget.PopupMenu(this, v)
            popup.menu.clear()

            if (currentCategory == Category.SLEEP) {
                popup.menu.add("今天")
                popup.menu.add("選擇日期")
            } else {
                popup.menu.add("近 7 天")
                popup.menu.add("近 30 天")
                popup.menu.add("全部")
                popup.menu.add("自訂區間")
            }

            popup.setOnMenuItemClickListener { item ->
                val title = item.title.toString()

                if (currentCategory == Category.SLEEP) {
                    when {
                        title == "今天" -> {
                            sleepDate = todayForSleep()
                            setDateNavVisible(true)
                            updateHeaderTime()
                            loadSleepDaily(isRefresh = true)
                        }
                        title.startsWith("選擇日期") -> { // 兼容「選擇日期…」
                            pickSleepDate() // 內部請記得 setDateNavVisible(true) + updateHeaderTime() + loadSleepDaily(true)
                        }
                    }
                    true
                } else {
                    when {
                        title == "近 7 天" -> setQuickRangeLastDays(7)
                        title == "近 30 天" -> setQuickRangeLastDays(30)
                        title == "全部" -> {
                            startTimeStr = null
                            endTimeStr = null
                            updateHeaderTime()
                            loadData(true)
                        }
                        title.startsWith("自訂區間") -> { // 兼容「自訂區間…」
                            pickCustomRangeDateOnly()
                        }
                    }
                    true
                }
            }

            popup.show()
        }

        spinnerCategory.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                when (position) {
                    0 -> {
                        setSleepBarVisible(false)
                        setDateNavVisible(false) // ← 隱藏
                        showEmpty("請先選擇類別")
                        swipe.isRefreshing = false
                    }
                    1 -> { // 跌倒列表
                        currentCategory = Category.FALL
                        setSleepBarVisible(false)
                        setDateNavVisible(false)                // ← 隱藏
                        recycler.adapter = adapter
                        if (!fallRangeInitialized) {
                            fallRangeInitialized = true
                            setQuickRangeLastDays(7)        // 這個函式會自己呼叫 loadData(true)
                        } else {
                            loadData(isRefresh = true)
                        }
                        updateHeaderTime()
                    }
                    2 -> { // 睡眠列表
                        currentCategory = Category.SLEEP
                        ensureSleepAdapter()
                        sleepDate = todayForSleep()
                        setDateNavVisible(true)      // ← 一進來就顯示日期列
                        updateSleepDateLabel()       // ← 顯示：日期：YYYY-MM-DD
                        loadSleepDaily(isRefresh = true)
                    }
                    3 -> { // 互動報告（占位）
                        currentCategory = Category.INTERACTION
                        setSleepBarVisible(false)
                        setDateNavVisible(false)      // ← 隱藏
                        recycler.visibility = View.GONE
                        progress.visibility = View.GONE
                        emptyState.visibility = View.VISIBLE
                        empty.text = "互動報告功能開發中"
                        swipe.isRefreshing = false
                        updateHeaderTime()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter = adapter

        swipe.setOnRefreshListener {
            when (spinnerCategory.selectedItemPosition) {
                1 -> loadData(true)            // 跌倒
                2 -> loadSleepDaily(true)      // 睡眠（daily）
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
            when (currentCategory) {
                Category.FALL -> { setDateNavVisible(false); loadData(true) }
                Category.SLEEP -> loadSleepDaily(true)
                else -> setDateNavVisible(false)
            }
        }
    }

    override fun onStop() {
        runCatching { unregisterReceiver(elderChangedReceiver) }
        super.onStop()
    }

    private fun ensureSleepAdapter() {
        if (recycler.adapter !== sleepAdapter) recycler.adapter = sleepAdapter
    }

    private fun loadSleepDaily(isRefresh: Boolean = false) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val elderId = sp.getInt(AppKeys.ELDER_ID, -1)

        if (elderId <= 0) { showEmpty("尚未選擇被照護者"); return }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        // 小工具：yyyy-MM-dd 前一天
        fun prevDayStr(ymd: String): String {
            val f = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
            val cal = java.util.Calendar.getInstance().apply { time = f.parse(ymd) ?: java.util.Date() }
            cal.add(java.util.Calendar.DAY_OF_MONTH, -1)
            return f.format(cal.time)
        }

        // 將 daily 的 data(單一物件) 轉成畫面用的單筆列表
        fun toRecordOrNull(data: com.example.myapplication.model.SleepDailyData?): com.example.myapplication.model.SleepRecord? {
            if (data == null) return null
            val st = data.sleepTime
            val wt = data.wakeTime
            if (st.isNullOrBlank() && wt.isNullOrBlank()) return null
            return com.example.myapplication.model.SleepRecord(
                sleepTime = st ?: wt ?: "",
                wakeTime  = wt ?: st ?: ""
            )
        }

        lifecycleScope.launch {
            try {
                // 先查選定日期
                val r1 = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getSleepDaily(elderId, sleepDate)
                }
                var item = toRecordOrNull(r1.data)

                // 若為空，再查「前一天」（應付以起床日/睡眠日分桶的情況）
                if (item == null) {
                    val alt = prevDayStr(sleepDate)
                    android.util.Log.d("SleepDaily", "fallback date=$alt")
                    val r2 = withContext(Dispatchers.IO) {
                        RetrofitClient.apiService.getSleepDaily(elderId, alt)
                    }
                    item = toRecordOrNull(r2.data)
                }

                swipe.isRefreshing = false
                if (item != null) {
                    ensureSleepAdapter()
                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                    sleepAdapter.submitList(listOf(item)) // 單一物件包成列表給 Adapter
                } else {
                    showEmpty("這天沒有睡眠紀錄（$sleepDate）")
                }
            } catch (_: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
            }
        }
    }

    private fun setDateNavVisible(visible: Boolean) {
        barDateNav.visibility = if (visible) View.VISIBLE else View.GONE
    }

    private fun updateSleepDateLabel() {
        // 睡眠列表的標題就直接用 textRange 呈現
        textRange.text = "日期：$sleepDate"
        textRange.visibility = View.VISIBLE
    }

    private fun toYmdOrNull(s: String?): String? {
        if (s.isNullOrBlank()) return null
        val patterns = listOf(
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd",
            "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        )
        for (p in patterns) try {
            val inFmt = SimpleDateFormat(p, Locale.getDefault())
            if (p.contains("'Z'")) inFmt.timeZone = TimeZone.getTimeZone("UTC")
            val d = inFmt.parse(s) ?: continue
            return SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(d)
        } catch (_: Exception) {}
        return s // 解析失敗就原樣回傳
    }

    private fun setHeaderForFall() {
        val start = toYmdOrNull(startTimeStr)
        val end   = toYmdOrNull(endTimeStr)
        val label = when {
            start == null && end == null -> "區間：全部"
            start != null && end != null -> "區間：$start ~ $end"
            start != null -> "區間：自 $start 起"
            else -> "區間：截至 $end"
        }
        textRange.text = label
        textRange.visibility = View.VISIBLE
    }

    private fun setHeaderForSleep() {
        textRange.text = "日期：$sleepDate"
        textRange.visibility = View.VISIBLE
    }

    private fun updateHeaderTime() {
        when (currentCategory) {
            Category.FALL -> setHeaderForFall()
            Category.SLEEP -> setHeaderForSleep()
            else -> textRange.visibility = View.GONE
        }
    }

    private fun refreshSleep() {
        when (sleepMode) {
            SleepMode.DAILY  -> loadSleepDaily(isRefresh = true)
            SleepMode.WEEKLY -> loadSleepWeekly(sleepDate, sundayFirst = true, isRefresh = true)
        }
    }

    private fun setSleepBarVisible(visible: Boolean) {
    }

    private fun todayForSleep(): String { // 若你已有就沿用
        val cal = Calendar.getInstance().apply { add(Calendar.HOUR_OF_DAY, -6) } // 晚睡友善
        val f = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        return f.format(cal.time)
    }

    private fun onSleepClick(item: SleepRecord) {
        val msg = buildString {
            appendLine("入睡：${item.sleepTime}")
            appendLine("起床：${item.wakeTime}")
            val dur = item.durationMin?.let { "${it/60}小時${it%60}分" }
            if (dur != null) appendLine("時長：$dur")
            if (!item.note.isNullOrBlank()) appendLine("備註：${item.note}")
        }
        androidXAlert(msg)
    }

    private fun today(): String {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        return f.format(Date())
    }
    private fun shiftSleepDate(days: Int) {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val cal = Calendar.getInstance()
        cal.time = f.parse(sleepDate) ?: Date()
        cal.add(Calendar.DAY_OF_MONTH, days)
        sleepDate = f.format(cal.time)
    }

    private fun pickSleepDate() {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val base = f.parse(sleepDate) ?: Date()
        val cal = Calendar.getInstance().apply { time = base }
        android.app.DatePickerDialog(
            this,
            { _, y, m, d ->
                val c = Calendar.getInstance().apply {
                    set(y, m, d, 0, 0, 0)
                    set(Calendar.MILLISECOND, 0)
                }
                sleepDate = f.format(c.time)
                updateSleepDateLabel()
                setDateNavVisible(true)             // ← 選過日期後才顯示左右鍵
                loadSleepDaily(true)
            },
            cal.get(Calendar.YEAR),
            cal.get(Calendar.MONTH),
            cal.get(Calendar.DAY_OF_MONTH)
        ).apply { setTitle("選擇日期") }.show()
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
                    RetrofitClient.apiService.getFallEventRecords(
                        userId = elderId,
                        startTime = startTimeStr,  // 可能為 null -> 後端就不套篩選
                        endTime   = endTimeStr
                    )
                }
                swipe.isRefreshing = false

                val records = resp.data?.records ?: emptyList()
                if (resp.success && records.isNotEmpty()) {
                    val sorted = records.sortedByDescending { parseEpoch(it.detected_time) }
                    fun normalizeTime(s: String) = s.replace('T', ' ').removeSuffix("Z")
                    val normalized = sorted.map { it.copy(detected_time = normalizeTime(it.detected_time)) }

                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                    adapter.submitList(normalized)
                } else {
                    val tip = buildString {
                        append("這位被照護者目前沒有跌倒事件")
                        if (startTimeStr != null || endTimeStr != null) {
                            append("（篩選區間：")
                            append(startTimeStr ?: "無起始")
                            append(" ~ ")
                            append(endTimeStr ?: "現在")
                            append("）")
                        }
                    }
                    showEmpty(tip)
                }
            } catch (_: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
            }
        }
    }

    private fun dayBoundaryString(dayMillis: Long, endOfDay: Boolean): String {
        val cal = java.util.Calendar.getInstance().apply {
            timeInMillis = dayMillis
            set(java.util.Calendar.HOUR_OF_DAY, if (endOfDay) 23 else 0)
            set(java.util.Calendar.MINUTE,      if (endOfDay) 59 else 0)
            set(java.util.Calendar.SECOND,      if (endOfDay) 59 else 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault())
        // 若後端吃 UTC，打開下一行：
        // sdf.timeZone = java.util.TimeZone.getTimeZone("UTC")
        return sdf.format(cal.time)
    }

    // 單純選日期的 DatePicker（不含時間）
    private fun pickDate(title: String, onPicked: (Long) -> Unit) {
        val now = java.util.Calendar.getInstance()
        android.app.DatePickerDialog(
            this,
            { _, y, m, d ->
                val cal = java.util.Calendar.getInstance().apply {
                    set(y, m, d, 0, 0, 0)
                    set(java.util.Calendar.MILLISECOND, 0)
                }
                onPicked(cal.timeInMillis)
            },
            now.get(java.util.Calendar.YEAR),
            now.get(java.util.Calendar.MONTH),
            now.get(java.util.Calendar.DAY_OF_MONTH)
        ).apply { setTitle(title) }.show()
    }

    // 「自訂區間…」→ 只選開始/結束日期
    private fun pickCustomRangeDateOnly() {
        pickDate("選擇開始日期") { startDay ->
            pickDate("選擇結束日期") { endDay ->
                if (endDay < startDay) {
                    Toast.makeText(this, "結束日期不可早於開始日期", Toast.LENGTH_SHORT).show()
                } else {
                    startTimeStr = dayBoundaryString(startDay, endOfDay = false) // 00:00:00
                    endTimeStr   = dayBoundaryString(endDay,   endOfDay = true)  // 23:59:59
                    updateHeaderTime()
                    loadData(isRefresh = true)
                }
            }
        }
    }

    private fun formatApi(millis: Long, useUtc: Boolean = false): String {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault())
        if (useUtc) sdf.timeZone = java.util.TimeZone.getTimeZone("UTC")
        return sdf.format(java.util.Date(millis))
    }

    private fun applyRange(startMillis: Long?, endMillis: Long?) {
        startTimeStr = startMillis?.let { formatApi(it) }
        endTimeStr   = endMillis?.let { formatApi(it) }
        loadData(isRefresh = true)
    }

    private fun setQuickRangeLastDays(days: Int) {
        // 迄日 = 今天；起日 = 往回 (days-1) 天
        val endCal = Calendar.getInstance() // 今天
        val startCal = Calendar.getInstance().apply {
            add(Calendar.DAY_OF_MONTH, -(days - 1))
        }
        startTimeStr = dayBoundaryString(startCal.timeInMillis, endOfDay = false) // 00:00:00
        endTimeStr   = dayBoundaryString(endCal.timeInMillis,   endOfDay = true)  // 23:59:59
        updateHeaderTime()
        loadData(isRefresh = true)
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

    private fun loadSleepWeekly(anchorDate: String = sleepDate, sundayFirst: Boolean = true, isRefresh: Boolean = false) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val elderId = sp.getInt(AppKeys.ELDER_ID, -1)
        if (elderId <= 0) { showEmpty("尚未選擇被照護者"); return }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        lifecycleScope.launch {
            try {
                val raw = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getSleepWeekly(elderId, anchorDate, sundayFirst).string()
                }
                android.util.Log.d("SleepWeekly", "raw=$raw")

                val items = parseSleepWeeklyRaw(raw)
                    .sortedBy { it.date }                    // 週內由小到大
                val records = items.map {
                    com.example.myapplication.model.SleepRecord(
                        sleepTime = it.sleepTime ?: "${it.date} 00:00:00",
                        wakeTime  = it.wakeTime  ?: "${it.date} 00:00:00"
                    )
                }.filter { it.sleepTime.isNotBlank() || it.wakeTime.isNotBlank() }

                swipe.isRefreshing = false
                if (records.isNotEmpty()) {
                    ensureSleepAdapter()
                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                    sleepAdapter.submitList(records)
                } else {
                    showEmpty("這週沒有睡眠紀錄（基準日：$anchorDate）")
                }
            } catch (_: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
            }
        }
    }

    // 放在 VideoListActivity 內或 Utils 檔都可
    data class SleepWeeklyItem(val date: String, val sleepTime: String?, val wakeTime: String?)

    private fun parseSleepWeeklyRaw(raw: String): List<SleepWeeklyItem> {
        val gson = Gson()
        val text = raw.trim()

        // 把一個 map 轉成資料列（同時相容 sleep_time / sleepTime）
        fun mapToItem(m: Map<String, Any?>): SleepWeeklyItem {
            val date = (m["date"] as? String).orEmpty()
            val st = (m["sleep_time"] as? String) ?: (m["sleepTime"] as? String)
            val wt = (m["wake_time"]  as? String) ?: (m["wakeTime"]  as? String)
            return SleepWeeklyItem(date, st, wt)
        }

        // A) 直接陣列: [ { date, sleep_time, wake_time }, ... ]
        if (text.startsWith("[")) {
            return try {
                val listType = object : TypeToken<List<Map<String, Any?>>>() {}.type
                val arr: List<Map<String, Any?>> = Gson().fromJson(text, listType)
                arr.map { mapToItem(it) }
            } catch (_: Exception) { emptyList() }
        }

        // B/C) 包裝物件: { success, data:[...] } 或 { success, data:{ "2025-07-13":{...}, ... } }
        data class Box(val data: Any?)
        val box = try { gson.fromJson(text, Box::class.java) } catch (_: Exception) { null }
        val data = box?.data ?: return emptyList()

        return when (data) {
            is List<*> -> {
                try {
                    val json = gson.toJson(data)
                    val listType = object : TypeToken<List<Map<String, Any?>>>() {}.type
                    val arr: List<Map<String, Any?>> = gson.fromJson(json, listType)
                    arr.map { mapToItem(it) }
                } catch (_: Exception) { emptyList() }
            }
            is Map<*, *> -> {
                try {
                    val json = gson.toJson(data)
                    val mapType = object : TypeToken<Map<String, Map<String, Any?>>>() {}.type
                    val m: Map<String, Map<String, Any?>> = gson.fromJson(json, mapType)
                    m.entries.map { (date, obj) ->
                        val st = (obj["sleep_time"] as? String) ?: (obj["sleepTime"] as? String)
                        val wt = (obj["wake_time"]  as? String) ?: (obj["wakeTime"]  as? String)
                        SleepWeeklyItem(date, st, wt)
                    }
                } catch (_: Exception) { emptyList() }
            }
            else -> emptyList()
        }
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
