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
import com.example.myapplication.model.SitRecordUi
import com.example.myapplication.adapter.SitRecordAdapter


class VideoListActivity : AppCompatActivity() {

    private lateinit var swipe: SwipeRefreshLayout
    private lateinit var recycler: RecyclerView
    private lateinit var progress: ProgressBar
    private lateinit var emptyState: View
    private lateinit var empty: TextView
    private val adapter by lazy { FallRecordAdapter(::onItemClick) }
    private enum class Category { FALL, SLEEP, INTERACTION, SIT }

    private lateinit var spinnerCategory: Spinner
    private var currentCategory: Category = Category.FALL
    private val sleepAdapter by lazy { com.example.myapplication.adapter.SleepRecordAdapter(::onSleepClick) }
    private val interactionAdapter by lazy {
        com.example.myapplication.adapter.InteractionReportAdapter(
            onPrevWeek = {
                shiftInteractionDate(-7)
                loadInteractionByWeek(interactionAnchorDate, false, isRefresh = true)
            },
            onNextWeek = {
                shiftInteractionDate(+7)
                loadInteractionByWeek(interactionAnchorDate, false, isRefresh = true)
            }
        )
    }

    private val sitAdapter by lazy { com.example.myapplication.adapter.SitRecordAdapter() }
    private var sitDate: String = todayForSleep()
    private enum class SitMode { DAILY, WEEKLY }
    private var sitMode: SitMode = SitMode.DAILY

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

    private lateinit var switchWeekly: com.google.android.material.materialswitch.MaterialSwitch
    private var sundayFirst: Boolean = true
    private var interactionAnchorDate: String = todayForSleep()

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

        switchWeekly = findViewById(R.id.switchWeekly)
        switchWeekly.visibility = View.GONE // 預設隱藏（只在睡眠頁顯示）
        switchWeekly.setOnCheckedChangeListener { _, isChecked ->
            when (currentCategory) {

                Category.SLEEP -> {
                    sleepMode = if (isChecked) SleepMode.WEEKLY else SleepMode.DAILY
                    updateHeaderTime()
                    if (sleepMode == SleepMode.WEEKLY) {
                        loadSleepWeekly(sleepDate, sundayFirst = sundayFirst, isRefresh = true)
                    } else {
                        loadSleepDaily(isRefresh = true)
                    }
                }

                Category.SIT -> {
                    sitMode = if (isChecked) SitMode.WEEKLY else SitMode.DAILY
                    updateHeaderTime()
                    if (sitMode == SitMode.WEEKLY) {
                        loadSitWeekly(sitDate, sundayFirst = sundayFirst, isRefresh = true)
                    } else {
                        loadSitDaily(isRefresh = true)
                    }
                }

                else -> {
                    // 其他分類不處理週/日切換
                }
            }
        }

        btnPrevDay.setOnClickListener {
            when (currentCategory) {
                Category.SLEEP -> {
                    sleepMode = if (switchWeekly.isChecked) SleepMode.WEEKLY else SleepMode.DAILY
                    shiftSleepDate(if (sleepMode == SleepMode.DAILY) -1 else -7)
                    updateHeaderTime()
                    refreshSleep()
                }
                Category.SIT -> {
                    sitMode = if (switchWeekly.isChecked) SitMode.WEEKLY else SitMode.DAILY
                    shiftSitDate(if (sitMode == SitMode.DAILY) -1 else -7)
                    updateHeaderTime()
                    if (sitMode == SitMode.WEEKLY) loadSitWeekly(sitDate, sundayFirst, isRefresh = true)
                    else loadSitDaily(isRefresh = true)
                }
                else -> Unit
            }
        }

        btnNextDay.setOnClickListener {
            when (currentCategory) {
                Category.SLEEP -> {
                    sleepMode = if (switchWeekly.isChecked) SleepMode.WEEKLY else SleepMode.DAILY
                    shiftSleepDate(if (sleepMode == SleepMode.DAILY) +1 else +7)
                    updateHeaderTime()
                    refreshSleep()
                }
                Category.SIT -> {
                    sitMode = if (switchWeekly.isChecked) SitMode.WEEKLY else SitMode.DAILY
                    shiftSitDate(if (sitMode == SitMode.DAILY) +1 else +7)
                    updateHeaderTime()
                    if (sitMode == SitMode.WEEKLY) loadSitWeekly(sitDate, sundayFirst, isRefresh = true)
                    else loadSitDaily(isRefresh = true)
                }
                else -> Unit
            }
        }

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        spinnerCategory = findViewById(R.id.spinnerCategory)

        val categories = listOf("— 選擇類別", "跌倒列表", "睡眠列表", "互動報告", "久坐資訊")
        val icons = listOf(0, R.drawable.ic_fall_24, R.drawable.ic_sleep_24, R.drawable.ic_report_24, R.drawable.ic_sit_24)

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

            when (currentCategory) {
                Category.SIT -> {
                    popup.menu.add("今天")
                    popup.menu.add("選擇日期")
                }
                Category.SLEEP -> {
                    popup.menu.add("今天")
                    popup.menu.add("選擇日期")
                }
                Category.INTERACTION -> {
                    popup.menu.add("最新")
                    popup.menu.add("選擇日期")
                }
                else -> { // FALL
                    popup.menu.add("近 7 天")
                    popup.menu.add("近 30 天")
                    popup.menu.add("全部")
                    popup.menu.add("自訂區間")
                }
            }

            popup.setOnMenuItemClickListener { item ->
                val title = item.title.toString()

                when (currentCategory) {

                    // ====== SLEEP ======
                    Category.SLEEP -> {
                        when (title) {
                            "今天" -> {
                                sleepDate = todayForSleep()
                                setDateNavVisible(true)
                                updateHeaderTime()
                                sleepMode = if (switchWeekly.isChecked) SleepMode.WEEKLY else SleepMode.DAILY
                                if (sleepMode == SleepMode.WEEKLY) {
                                    loadSleepWeekly(sleepDate, sundayFirst = sundayFirst, isRefresh = true)
                                } else {
                                    loadSleepDaily(isRefresh = true)
                                }
                            }
                            "選擇日期" -> {
                                // 等 popup 關閉後再跳 DatePicker
                                popup.setOnDismissListener { pickSleepDateRetainMode() }
                                popup.dismiss()
                            }
                        }
                        true
                    }

                    // ====== INTERACTION（互動報告）======
                    Category.INTERACTION -> {
                        when (title) {
                            "最新" -> {
                                loadInteractionLatest(isRefresh = true) // 改成打 latest API
                            }
                            "選擇日期" -> {
                                popup.setOnDismissListener { pickInteractionDate() }
                                popup.dismiss()
                            }
                        }
                        true
                    }

                    // ====== SIT ======
                    Category.SIT -> {
                        when (title) {
                            "今天" -> {
                                sitDate = todayForSleep()
                                setDateNavVisible(true)
                                updateHeaderTime()
                                sitMode = if (switchWeekly.isChecked) SitMode.WEEKLY else SitMode.DAILY
                                if (sitMode == SitMode.WEEKLY) loadSitWeekly(sitDate, sundayFirst, true)
                                else loadSitDaily(true)
                            }
                            "選擇日期" -> {
                                popup.setOnDismissListener { pickSitDateRetainMode() }
                                popup.dismiss()
                            }
                        }
                        true
                    }

                    // ====== FALL ======
                    else -> {
                        when (title) {
                            "近 7 天"  -> setQuickRangeLastDays(7)
                            "近 30 天" -> setQuickRangeLastDays(30)
                            "全部"     -> { startTimeStr = null; endTimeStr = null; updateHeaderTime(); loadData(true) }
                            "自訂區間" -> {
                                popup.setOnDismissListener { pickCustomRangeDateOnly() }
                                popup.dismiss()
                            }
                        }
                        true
                    }
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
                        setDateNavVisible(true)
                        switchWeekly.visibility = View.VISIBLE
                        switchWeekly.isChecked = false       // 預設單日；要記憶上次可改成讀 SP
                        sleepMode = SleepMode.DAILY
                        updateHeaderTime()
                        loadSleepDaily(isRefresh = true)
                    }
                    3 -> {
                        currentCategory = Category.INTERACTION
                        setSleepBarVisible(false)
                        setDateNavVisible(false)
                        recycler.adapter = interactionAdapter

                        progress.visibility = View.VISIBLE
                        recycler.visibility = View.GONE
                        emptyState.visibility = View.GONE

                        interactionAnchorDate = todayForSleep() // yyyy-MM-dd
                        loadInteractionByWeek(interactionAnchorDate, /* sundayFirstDay = */ false)
                    }
                    4 -> { // 久坐
                        currentCategory = Category.SIT
                        recycler.adapter = sitAdapter
                        sitDate = todayForSleep()
                        setDateNavVisible(true)               // 共用左右切換
                        switchWeekly.visibility = View.VISIBLE
                        switchWeekly.isChecked = false
                        sitMode = SitMode.DAILY
                        updateHeaderTime()
                        loadSitDaily(isRefresh = true)
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter = adapter

        swipe.setOnRefreshListener {
            when (spinnerCategory.selectedItemPosition) {
                1 -> loadData(true)
                2 -> refreshSleep()
                3 -> loadInteractionByWeek(interactionAnchorDate, false, true)
                4 -> {
                    if (sitMode == SitMode.WEEKLY) loadSitWeekly(sitDate, sundayFirst, true)
                    else loadSitDaily(true)
                }
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
                Category.FALL -> {
                    setDateNavVisible(false)
                    loadData(isRefresh = true)
                }
                Category.SLEEP -> {
                    refreshSleep()
                }
                Category.INTERACTION -> {
                    loadInteractionByWeek(interactionAnchorDate, /* sundayFirstDay = */ false, isRefresh = true)
                }
                Category.SIT -> {
                    // 依目前切換狀態（日/週）重載久坐資料
                    if (sitMode == SitMode.WEEKLY) {
                        loadSitWeekly(sitDate, sundayFirst, isRefresh = true)
                    } else {
                        loadSitDaily(isRefresh = true)
                    }
                }
            }
        }
    }

    override fun onStop() {
        runCatching { unregisterReceiver(elderChangedReceiver) }
        super.onStop()
    }

    private fun shiftSleepDate(days: Int) {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val cal = Calendar.getInstance()
        cal.time = f.parse(sleepDate) ?: Date()
        cal.add(Calendar.DAY_OF_MONTH, days)
        sleepDate = f.format(cal.time)
    }

    private fun shiftSitDate(days: Int) {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val cal = Calendar.getInstance()
        cal.time = f.parse(sitDate) ?: Date()
        cal.add(Calendar.DAY_OF_MONTH, days)
        sitDate = f.format(cal.time)
    }

    private fun pickSitDateRetainMode() {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val base = f.parse(sitDate) ?: Date()
        val cal = Calendar.getInstance().apply { time = base }
        android.app.DatePickerDialog(
            this,
            { _, y, m, d ->
                val c = Calendar.getInstance().apply { set(y, m, d, 0,0,0); set(Calendar.MILLISECOND, 0) }
                sitDate = f.format(c.time)
                setDateNavVisible(true)
                updateHeaderTime()
                sitMode = if (switchWeekly.isChecked) SitMode.WEEKLY else SitMode.DAILY
                if (sitMode == SitMode.WEEKLY) loadSitWeekly(sitDate, sundayFirst, true)
                else loadSitDaily(true)
            },
            cal.get(Calendar.YEAR),
            cal.get(Calendar.MONTH),
            cal.get(Calendar.DAY_OF_MONTH)
        ).apply { setTitle("選擇日期") }.show()
    }

    private fun loadSitDaily(isRefresh: Boolean = false) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val userId = sp.getInt(AppKeys.ELDER_ID, -1)
        if (userId <= 0) { showEmpty("尚未選擇被照護者"); return }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        lifecycleScope.launch {
            try {
                val box = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getSitDaily(userId, sitDate)
                }

                val recs = box.data?.records ?: emptyList()
                if (box.success && recs.isNotEmpty()) {
                    val list = recs.map { r ->
                        val date = (r.startAt ?: r.endAt ?: "").take(10) // yyyy-MM-dd
                        SitRecordUi(
                            dateLabel = date,                    // 單日也顯示日期
                            startAt = r.startAt.orEmpty(),       // 交給 Adapter 只顯示時間
                            endAt   = r.endAt.orEmpty(),
                            durationLabel = formatDuration(r.startAt, r.endAt)
                        )
                    }
                    recycler.adapter = sitAdapter
                    sitAdapter.showDate = true                  // 單日顯示日期列
                    sitAdapter.submitList(list)

                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                } else {
                    showEmpty("這天沒有久坐紀錄（$sitDate）")
                }

            } catch (e: retrofit2.HttpException) {
                // 204/404 視為沒有資料，其餘才當伺服器錯誤
                if (e.code() == 204 || e.code() == 404) {
                    showEmpty("這天沒有久坐紀錄（$sitDate）")
                } else {
                    showEmpty("伺服器錯誤 ${e.code()}，請稍後再試")
                }

            } catch (_: java.io.IOException) {
                showEmpty("網路未連線或逾時")

            } catch (_: Exception) {
                showEmpty("連線失敗，請重試")

            } finally {
                swipe.isRefreshing = false
            }
        }
    }

    private fun normalize(s: String?): String = s?.replace('T',' ')?.removeSuffix("Z") ?: ""

    private fun formatDuration(start: String?, end: String?): String {
        val p = listOf(
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        )
        fun parse(x: String?): Long? {
            if (x.isNullOrBlank()) return null
            for (fmt in p) try {
                val sdf = SimpleDateFormat(fmt, Locale.getDefault())
                return sdf.parse(x)?.time
            } catch (_: Exception) {}
            return null
        }
        val s = parse(start); val e = parse(end)
        if (s == null || e == null || e < s) return "—"
        val min = ((e - s) / 60000).toInt()
        val h = min / 60; val m = min % 60
        return if (h > 0) "${h}小時${m}分" else "${m}分"
    }

    private fun loadSitWeekly(
        date: String = sitDate,
        sundayFirst: Boolean,
        isRefresh: Boolean = false
    ) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val userId = sp.getInt(AppKeys.ELDER_ID, -1)
        if (userId <= 0) { showEmpty("尚未選擇被照護者"); return }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        lifecycleScope.launch {
            try {
                val box = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getSitWeekly(userId, date, sundayFirst)
                }

                val recs = box.data?.records ?: emptyList()
                if (box.success && recs.isNotEmpty()) {
                    val list = recs.map { r ->
                        val d = (r.startAt ?: r.endAt ?: "").take(10)
                        SitRecordUi(
                            dateLabel = d,
                            startAt = normalize(r.startAt),
                            endAt   = normalize(r.endAt),
                            durationLabel = formatDuration(r.startAt, r.endAt)
                        )
                    }
                    recycler.adapter = sitAdapter
                    sitAdapter.showDate = true
                    sitAdapter.submitList(list)
                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                } else {
                    val (st, ed) = weekRangeOf(date, sundayFirst)
                    showEmpty("這週沒有久坐紀錄\n（${noBreakRangeLabel(compactRange(st, ed))}）")
                    empty.textAlignment = View.TEXT_ALIGNMENT_CENTER
                }
            } catch (e: retrofit2.HttpException) {
                if (e.code() in listOf(204, 404)) {
                    val (st, ed) = weekRangeOf(date, sundayFirst)
                    showEmpty("這週沒有久坐紀錄\n（${noBreakRangeLabel(compactRange(st, ed))}）")
                    empty.textAlignment = View.TEXT_ALIGNMENT_CENTER
                } else {
                    showEmpty("伺服器錯誤 ${e.code()}，請稍後再試")
                }
            } catch (_: java.io.IOException) {
                showEmpty("網路未連線或逾時")
            } catch (_: Exception) {
                showEmpty("連線失敗，請重試")
            } finally {
                swipe.isRefreshing = false
            }
        }
    }

    private fun shiftInteractionDate(days: Int) {
        val f = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val cal = java.util.Calendar.getInstance()
        cal.time = f.parse(interactionAnchorDate) ?: java.util.Date()
        cal.add(java.util.Calendar.DAY_OF_MONTH, days)
        interactionAnchorDate = f.format(cal.time)
    }


    private fun ensureSleepAdapter() {
        if (recycler.adapter !== sleepAdapter) recycler.adapter = sleepAdapter
    }

    private fun loadInteractionLatest(isRefresh: Boolean = false) {
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
                val box = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getWeeklyReportLatest(elderId)
                }
                swipe.isRefreshing = false
                bindInteractionUiFromBox(box, todayForSleep()) // 你已有此方法
            } catch (e: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
                android.util.Log.e("Interaction", "latest failed", e)
            }
        }
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
                    sleepAdapter.showDate = false
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

    private fun pickInteractionDate() {
        val f = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val base = f.parse(interactionAnchorDate) ?: java.util.Date()
        val cal = java.util.Calendar.getInstance().apply { time = base }

        android.app.DatePickerDialog(
            this,
            { _, y, m, d ->
                val c = java.util.Calendar.getInstance().apply {
                    set(y, m, d, 0, 0, 0)
                    set(java.util.Calendar.MILLISECOND, 0)
                }
                interactionAnchorDate = f.format(c.time)
                loadInteractionByWeek(interactionAnchorDate, false, isRefresh = true)
            },
            cal.get(java.util.Calendar.YEAR),
            cal.get(java.util.Calendar.MONTH),
            cal.get(java.util.Calendar.DAY_OF_MONTH)
        ).show()  // ← 不要 setTitle()
    }

    private fun setDateNavVisible(visible: Boolean) {
        barDateNav.visibility = if (visible) View.VISIBLE else View.GONE
        val showSwitch = visible && (currentCategory == Category.SLEEP || currentCategory == Category.SIT)
        switchWeekly.visibility = if (showSwitch) View.VISIBLE else View.GONE
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
        if (sleepMode == SleepMode.DAILY) {
            textRange.text = "日期：\n$sleepDate"
        } else {
            val (st, ed) = weekRangeOf(sleepDate, sundayFirst)
            val compact = compactRange(st, ed)           // 例如：2025-07-13 ~ 07-19
            val safe    = noBreakRangeLabel(compact)     // 轉成不可拆的版本
            textRange.text = "週期：\n$safe"              // 上下兩行
        }
        textRange.visibility = View.VISIBLE
    }

    private fun updateHeaderTime() {
        when (currentCategory) {
            Category.FALL         -> setHeaderForFall()
            Category.SLEEP        -> setHeaderForSleep()
            Category.SIT          -> setHeaderForSit()        // ← 新增
            Category.INTERACTION  -> textRange.visibility = View.GONE
        }
    }

    private fun setHeaderForSit() {
        // 跟 setHeaderForSleep 相同邏輯：日模式顯示單日；週模式顯示「週期：\n起訖」
        if (sitMode == SitMode.DAILY) {
            textRange.text = "日期：\n$sitDate"
        } else {
            val (st, ed) = weekRangeOf(sitDate, sundayFirst)
            val compact  = compactRange(st, ed)         // 例如：2025-09-01 ~ 09-07
            val safe     = noBreakRangeLabel(compact)   // 不斷行處理
            textRange.text = "週期：\n$safe"
        }
        textRange.visibility = View.VISIBLE
    }

    private fun loadInteractionByWeek(
        date: String = interactionAnchorDate,
        sundayFirstDay: Boolean = false,
        isRefresh: Boolean = false
    ) {
        val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
        val elderId = sp.getInt(AppKeys.ELDER_ID, -1)
        if (elderId <= 0) { showEmpty("尚未選擇被照護者"); return }

        if (!isRefresh) {
            progress.visibility = View.VISIBLE
            recycler.visibility = View.GONE
            emptyState.visibility = View.GONE
        }

        // 工具：從 JsonObject 取字串，兼容 snake/camel
        fun pickStr(o: com.google.gson.JsonObject, snake: String, camel: String): String {
            val a = if (o.has(snake) && !o.get(snake).isJsonNull) o.get(snake).asString else null
            val b = if (o.has(camel) && !o.get(camel).isJsonNull) o.get(camel).asString else null
            return a ?: b ?: ""
        }

        // 工具：把 JsonObject 轉成 WeeklyReportDto
        fun toDto(obj: com.google.gson.JsonObject): com.example.myapplication.model.WeeklyReportDto {
            val start = pickStr(obj, "start_date", "startDate")
            val end   = pickStr(obj, "end_date",   "endDate")
            val text  = pickStr(obj, "analysis_result", "content")
            return com.example.myapplication.model.WeeklyReportDto(start, end, text)
        }

        // 綁 UI
        fun bindFromDto(dto: com.example.myapplication.model.WeeklyReportDto, anchor: String) {
            val start = dto.startDate.orEmpty()
            val end   = dto.endDate.orEmpty()
            val text  = dto.content.orEmpty()
            val period = "週期：\n${start.replace('T',' ').replace("Z","")} ~ ${end.replace('T',' ').replace("Z","")}"

            progress.visibility = View.GONE
            emptyState.visibility = View.GONE
            recycler.visibility = View.VISIBLE
            recycler.adapter = interactionAdapter
            interactionAdapter.submit(
                com.example.myapplication.adapter.InteractionWeeklyUi(
                    periodLabel = period,
                    content = text
                )
            )
            interactionAnchorDate = end.take(10).ifBlank { anchor }
        }

        lifecycleScope.launch {
            try {
                android.util.Log.d("Interaction", "by-week req elderId=$elderId date=$date sundayFirst=$sundayFirstDay")

                val box = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getWeeklyReportByWeek(
                        elderId = elderId,
                        date = date,
                        sundayFirst = sundayFirstDay
                    )
                }
                swipe.isRefreshing = false
                android.util.Log.d("Interaction", "by-week resp success=${box.success} code=${box.code} msg=${box.message}")

                val elem = box.data
                if (!box.success || elem == null || elem.isJsonNull) {
                    showEmpty(box.message ?: "這一週沒有互動報告"); return@launch
                }

                // data 可能是 Object 或 Array：先取第一個物件
                val obj: com.google.gson.JsonObject? = when {
                    elem.isJsonObject -> elem.asJsonObject
                    elem.isJsonArray  -> elem.asJsonArray.firstOrNull()?.asJsonObject
                    else -> null
                }

                val dto = obj?.let { toDto(it) }
                if (dto != null) bindFromDto(dto, date)
                else showEmpty("這一週沒有互動報告")

            } catch (e: retrofit2.HttpException) {
                swipe.isRefreshing = false
                if (e.code() == 500) {
                    // 退回 latest 撐畫面（latest 回傳固定是物件）
                    try {
                        val latest = withContext(Dispatchers.IO) {
                            RetrofitClient.apiService.getWeeklyReportLatest(elderId)
                        }
                        latest.data?.let { bindFromDto(it, date) }
                            ?: showEmpty("伺服器錯誤 500")
                    } catch (e2: Exception) {
                        showEmpty("伺服器錯誤 500，且最新一週也取得失敗")
                        android.util.Log.e("Interaction", "latest fallback failed", e2)
                    }
                } else {
                    showEmpty("伺服器錯誤 ${e.code()}，請重試")
                }
            } catch (e: java.io.IOException) {
                swipe.isRefreshing = false
                showEmpty("網路未連線或逾時")
            } catch (e: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
                android.util.Log.e("Interaction", "by-week failed", e)
            }
        }
    }

    private fun bindInteractionUiFromBox(box: com.example.myapplication.model.WeeklyReportBox, anchor: String) {
        val data = box.data
        if (box.success && data != null) {
            val start = data.startDate.orEmpty()
            val end   = data.endDate.orEmpty()
            val text  = data.content.orEmpty()
            val period = "週期：\n${start.replace('T',' ').replace("Z","")} ~ ${end.replace('T',' ').replace("Z","")}"

            progress.visibility = View.GONE
            emptyState.visibility = View.GONE
            recycler.visibility = View.VISIBLE
            recycler.adapter = interactionAdapter
            interactionAdapter.submit(
                com.example.myapplication.adapter.InteractionWeeklyUi(period, text)
            )
            interactionAnchorDate = end.take(10).ifBlank { anchor }
        } else {
            showEmpty(box.message ?: "目前沒有每週互動報告")
        }
    }

    private fun formatWeekLabel(st: String, ed: String): String {
        val inFmt = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val d1 = runCatching { inFmt.parse(st) }.getOrNull()
        val d2 = runCatching { inFmt.parse(ed) }.getOrNull()
        if (d1 == null || d2 == null) return "$st~$ed"

        val y1 = Calendar.getInstance().apply { time = d1 }.get(Calendar.YEAR)
        val y2 = Calendar.getInstance().apply { time = d2 }.get(Calendar.YEAR)

        val startFmt = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val endFmt   = if (y1 == y2) SimpleDateFormat("MM-dd", Locale.getDefault())
        else SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())

        return "${startFmt.format(d1)} ~ ${endFmt.format(d2)}"
    }

    private fun refreshSleep() {
        when (sleepMode) {
            SleepMode.DAILY  -> loadSleepDaily(isRefresh = true)
            SleepMode.WEEKLY -> loadSleepWeekly(sleepDate, sundayFirst = sundayFirst, isRefresh = true)
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
    private fun shiftDate(mut: () -> String, days: Int) {
        val f = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val cal = Calendar.getInstance()
        cal.time = f.parse(sleepDate) ?: Date()
        cal.add(Calendar.DAY_OF_MONTH, days)
        sleepDate = f.format(cal.time)
    }

    private fun pickSleepDateRetainMode() {
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
                setDateNavVisible(true)
                updateHeaderTime()

                // 依開關狀態切換日 or 週
                sleepMode = if (switchWeekly.isChecked) SleepMode.WEEKLY else SleepMode.DAILY
                if (sleepMode == SleepMode.WEEKLY) {
                    loadSleepWeekly(sleepDate, sundayFirst = sundayFirst, isRefresh = true)
                } else {
                    loadSleepDaily(isRefresh = true)
                }
            },
            cal.get(Calendar.YEAR),
            cal.get(Calendar.MONTH),
            cal.get(Calendar.DAY_OF_MONTH)
        ).apply { setTitle("選擇日期") }.show()
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

    private fun loadSleepWeekly(
        anchorDate: String = sleepDate,
        sundayFirst: Boolean = true,
        isRefresh: Boolean = false
    ) {
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
                val box = withContext(Dispatchers.IO) {
                    RetrofitClient.apiService.getSleepWeeklyBox(elderId, anchorDate, sundayFirst)
                }

                // ---- 標準化成 List<SleepRecord> ----
                val records: List<SleepRecord> = when {
                    box.data == null || box.data.isJsonNull -> emptyList()
                    box.data.isJsonArray -> emptyList() // [] → 尚無資料
                    box.data.isJsonObject -> {
                        val payload = Gson().fromJson(
                            box.data,
                            com.example.myapplication.model.WeeklySleepPayload::class.java
                        )
                        val raw = payload.records ?: emptyList()
                        raw.map { r ->
                            SleepRecord(
                                sleepTime = r.sleepTime.orEmpty(),
                                wakeTime  = r.wakeTime.orEmpty()
                            )
                        }.filter { it.sleepTime.isNotBlank() || it.wakeTime.isNotBlank() }
                    }
                    else -> emptyList()
                }

                swipe.isRefreshing = false

                if (records.isNotEmpty()) {
                    ensureSleepAdapter()
                    sleepAdapter.showDate = true            // 週模式顯示日期
                    progress.visibility = View.GONE
                    emptyState.visibility = View.GONE
                    recycler.visibility = View.VISIBLE
                    sleepAdapter.submitList(records)
                } else {
                    // 沒資料 → 置中、兩行（第二行顯示壓縮區間）
                    val (st, ed) = weekRangeOf(anchorDate, sundayFirst)
                    val compact  = compactRange(st, ed)         // 例：2025-07-20 ~ 07-26
                    val safe     = noBreakRangeLabel(compact)   // 防斷行：2011/nbsp

                    recycler.visibility = View.GONE
                    progress.visibility = View.GONE
                    emptyState.visibility = View.VISIBLE
                    empty.text = "這週沒有睡眠紀錄\n（$safe）"
                    empty.textAlignment = View.TEXT_ALIGNMENT_CENTER
                }
            } catch (_: Exception) {
                swipe.isRefreshing = false
                showEmpty("連線失敗，請重試")
            }
        }
    }

    // 例：同年 -> "2025-07-13 ~ 07-19"；跨年 -> "2025-12-30 ~ 2026-01-05"
    private fun compactRange(start: String, end: String): String {
        val inFmt = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val s = runCatching { inFmt.parse(start) }.getOrNull()
        val e = runCatching { inFmt.parse(end) }.getOrNull()
        if (s == null || e == null) return "$start ~ $end"

        val cs = java.util.Calendar.getInstance().apply { time = s }
        val ce = java.util.Calendar.getInstance().apply { time = e }
        val sameYear = cs.get(java.util.Calendar.YEAR) == ce.get(java.util.Calendar.YEAR)

        val sFmt = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val eFmt = java.text.SimpleDateFormat(if (sameYear) "MM-dd" else "yyyy-MM-dd", java.util.Locale.getDefault())
        return "${sFmt.format(s)} ~ ${eFmt.format(e)}"
    }

    // 防止日期在 - 或 ~ 兩側被拆行
    private fun noBreakRangeLabel(range: String): String {
        return range
            .replace("-", "\u2011")          // 不斷行連字號 (non-breaking hyphen)
            .replace(" ~ ", "\u00A0~\u00A0") // 不斷行空白 NBSP
    }

    // === 計算 anchorDate 所在週的起訖（回傳 kotlin.Pair）===
    private fun weekRangeOf(anchorYmd: String, sundayFirst: Boolean): kotlin.Pair<String, String> {
        val f = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        val cal = java.util.Calendar.getInstance().apply {
            time = f.parse(anchorYmd) ?: java.util.Date()
            set(java.util.Calendar.HOUR_OF_DAY, 0)
            set(java.util.Calendar.MINUTE, 0)
            set(java.util.Calendar.SECOND, 0)
            set(java.util.Calendar.MILLISECOND, 0)
        }

        val dow = cal.get(java.util.Calendar.DAY_OF_WEEK) // Sun=1 .. Sat=7
        val offsetToStart = if (sundayFirst) {
            // 週日為 0 位移
            -((dow - java.util.Calendar.SUNDAY + 7) % 7)
        } else {
            // 週一為 0 位移
            -((dow - java.util.Calendar.MONDAY + 7) % 7)
        }

        cal.add(java.util.Calendar.DAY_OF_MONTH, offsetToStart)
        val start = f.format(cal.time)

        cal.add(java.util.Calendar.DAY_OF_MONTH, 6)
        val end = f.format(cal.time)

        return kotlin.Pair(start, end)
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
