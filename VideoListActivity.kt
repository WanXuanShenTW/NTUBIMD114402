package com.example.myapplication

import android.app.DatePickerDialog
import android.content.Context
import android.content.Intent
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.VideoEventAdapter
import com.example.myapplication.model.FavoriteRequest
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.util.*

class VideoListActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: VideoEventAdapter
    private lateinit var editStartDate: EditText
    private lateinit var editEndDate: EditText
    private lateinit var btnSearch: Button
    private lateinit var loadingProgress: ProgressBar
    private var isFavoriteActionRunning = false
    private var isDataLoaded = false

    private val apiService = RetrofitClient.apiService
    private var caregiverId: Int = -1
    private var elderId: String? = null  // 改為 String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_list)

        recyclerView = findViewById(R.id.recyclerView)
        editStartDate = findViewById(R.id.editStartDate)
        editEndDate = findViewById(R.id.editEndDate)
        btnSearch = findViewById(R.id.btnSearch)
        loadingProgress = findViewById(R.id.loadingProgress)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener { finish() }

        val sharedPref = getSharedPreferences("smartcare_pref", Context.MODE_PRIVATE)
        caregiverId = sharedPref.getInt("user_id", -1)
        elderId = sharedPref.getString("elder_id", null)

        if (elderId.isNullOrEmpty()) {
            Toast.makeText(this, "請先選擇被照護者", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        adapter = VideoEventAdapter(
            mutableListOf(),
            onItemClick = { event ->
                val videoUrl = "${ApiConfig.BASE_URL}fall_video_file?record_id=${event.record_id}"
                val intent = Intent(this, VideoPlayerActivity::class.java).apply {
                    putExtra("record_id", event.record_id)
                    putExtra("video_url", videoUrl)
                }
                startActivity(intent)
            },
            onFavoriteClick = { event, _, done ->
                if (!isDataLoaded) {
                    Toast.makeText(this, "資料尚未載入完成", Toast.LENGTH_SHORT).show()
                    return@VideoEventAdapter
                }
                if (isFavoriteActionRunning) {
                    Toast.makeText(this, "正在處理收藏中，請稍後", Toast.LENGTH_SHORT).show()
                    return@VideoEventAdapter
                }

                isFavoriteActionRunning = true

                if (event.isFavorite) {
                    removeFromFavorite(event) { success ->
                        if (success) {
                            event.isFavorite = false
                            adapter.updateFavoriteByRecordId(event.record_id, event.isFavorite)
                        } else {
                            Toast.makeText(this, "取消收藏失敗", Toast.LENGTH_SHORT).show()
                        }
                        isFavoriteActionRunning = false
                        done()
                    }
                } else {
                    addToFavorite(event) { success ->
                        if (success) {
                            event.isFavorite = true
                            adapter.updateFavoriteByRecordId(event.record_id, event.isFavorite)
                        } else {
                            Toast.makeText(this, "加入收藏失敗", Toast.LENGTH_SHORT).show()
                        }
                        isFavoriteActionRunning = false
                        done()
                    }
                }
            }
        )
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        val today = Calendar.getInstance()
        editEndDate.setText(String.format("%04d-%02d-%02d",
            today.get(Calendar.YEAR), today.get(Calendar.MONTH) + 1, today.get(Calendar.DAY_OF_MONTH)))

        fun showDatePicker(onDateSelected: (String) -> Unit) {
            val calendar = Calendar.getInstance()
            DatePickerDialog(this,
                { _, year, month, day ->
                    val dateStr = String.format("%04d-%02d-%02d", year, month + 1, day)
                    onDateSelected(dateStr)
                },
                calendar.get(Calendar.YEAR),
                calendar.get(Calendar.MONTH),
                calendar.get(Calendar.DAY_OF_MONTH)
            ).show()
        }

        editStartDate.setOnClickListener { showDatePicker { editStartDate.setText(it) } }
        editEndDate.setOnClickListener { showDatePicker { editEndDate.setText(it) } }

        btnSearch.setOnClickListener { queryVideoList() }
    }

    private fun queryVideoList() {
        val startDate = editStartDate.text.toString()
        val endDate = editEndDate.text.toString()

        if (startDate.isBlank() || endDate.isBlank()) {
            Toast.makeText(this, "請選擇開始與結束日期", Toast.LENGTH_SHORT).show()
            return
        }

        isDataLoaded = false
        loadingProgress.visibility = View.VISIBLE
        recyclerView.visibility = View.GONE

        apiService.getFallVideos(caregiverId, elderId!!, startDate, endDate, limit = 5)
            .enqueue(object : Callback<List<VideoEvent>> {
                override fun onResponse(call: Call<List<VideoEvent>>, response: Response<List<VideoEvent>>) {
                    if (response.isSuccessful && response.body() != null) {
                        val events = response.body()!!.map {
                            it.copy(
                                isFavorite = it.in_watchlist,
                                video_type = it.video_type ?: "fall"
                            )
                        }.toMutableList()
                        adapter.updateData(events)
                        isDataLoaded = true
                    } else {
                        Toast.makeText(this@VideoListActivity, "查詢失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                    loadingProgress.visibility = View.GONE
                    recyclerView.visibility = View.VISIBLE
                }
                override fun onFailure(call: Call<List<VideoEvent>>, t: Throwable) {
                    Toast.makeText(this@VideoListActivity, "連線錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                    loadingProgress.visibility = View.GONE
                    recyclerView.visibility = View.VISIBLE
                }
            })
    }

    private fun addToFavorite(event: VideoEvent, onFinish: (Boolean) -> Unit) {
        val data = FavoriteRequest(
            user_id = caregiverId,
            record_id = event.record_id,
            video_type = event.video_type ?: "fall"
        )

        Log.d("FavoriteDebug", "發送 JSON = user_id=${data.user_id}, record_id=${data.record_id}, video_type=${data.video_type}")

        apiService.addFavorite(data).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                Log.d("FavoriteDebug", "addFavorite HTTP=${response.code()} body=${response.body()} err=${response.errorBody()?.string()}")
                onFinish(response.isSuccessful)
            }
            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Log.e("FavoriteDebug", "加入收藏錯誤：${t.message}")
                onFinish(false)
            }
        })
    }

    private fun removeFromFavorite(event: VideoEvent, onFinish: (Boolean) -> Unit) {
        val data = FavoriteRequest(
            user_id = caregiverId,
            record_id = event.record_id,
            video_type = event.video_type ?: "fall"
        )
        apiService.removeFavorite(data).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                Log.d("FavoriteDebug", "removeFavorite HTTP=${response.code()} body=${response.body()} err=${response.errorBody()?.string()}")
                onFinish(response.isSuccessful)
            }
            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Log.e("FavoriteDebug", "取消收藏錯誤：${t.message}")
                onFinish(false)
            }
        })
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.action == MotionEvent.ACTION_DOWN) {
            val v = currentFocus
            if (v is EditText) {
                val outRect = Rect()
                v.getGlobalVisibleRect(outRect)
                if (!outRect.contains(ev.rawX.toInt(), ev.rawY.toInt())) {
                    v.clearFocus()
                    hideKeyboard(v)
                }
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun hideKeyboard(view: View) {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }
}
