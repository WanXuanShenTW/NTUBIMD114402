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
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.model.FavoriteRequest
import com.example.myapplication.network.RetrofitClient
import com.example.myapplication.adapter.VideoEventAdapter
import retrofit2.*
import java.util.*

class VideoListActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: VideoEventAdapter
    private lateinit var editUserId: EditText
    private lateinit var editStartDate: EditText
    private lateinit var editEndDate: EditText
    private lateinit var btnSearch: Button
    private lateinit var loadingProgress: ProgressBar

    private val apiService = RetrofitClient.apiService
    private var userId: Int = -1
    private var isDataLoaded = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_list)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener {
            startActivity(Intent(this, DashboardActivity::class.java))
            finish()
        }

        recyclerView = findViewById(R.id.recyclerView)
        editUserId = findViewById(R.id.editUserId)
        editStartDate = findViewById(R.id.editStartDate)
        editEndDate = findViewById(R.id.editEndDate)
        btnSearch = findViewById(R.id.btnSearch)
        loadingProgress = findViewById(R.id.loadingProgress)

        userId = 529
        editUserId.setText(userId.toString())
        editUserId.isEnabled = false

        adapter = VideoEventAdapter(
            mutableListOf(),
            onItemClick = { event ->
                val videoUrl = "https://c8fd-60-250-79-110.ngrok-free.app/fall_video_file?record_id=${event.record_id}"
                val intent = Intent(this, VideoPlayerActivity::class.java).apply {
                    putExtra("record_id", event.record_id) // 🔥 加這行！
                    putExtra("video_url", videoUrl)
                    putExtra("video_filename", event.video_filename)
                }
                startActivity(intent)
            },
            onFavoriteClick = { event ->
                if (!isDataLoaded) {
                    Toast.makeText(this, "資料尚未載入完成，請稍後再試", Toast.LENGTH_SHORT).show()
                    return@VideoEventAdapter
                }
                if (event.isFavorite) {
                    removeFromFavorite(event)
                } else {
                    addToFavorite(event)
                }
                event.isFavorite = !event.isFavorite
                adapter.notifyItemChanged(adapter.getItemPosition(event))
            }
        )

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        fun showDatePicker(onDateSelected: (String) -> Unit) {
            val calendar = Calendar.getInstance()
            val datePicker = DatePickerDialog(
                this,
                { _, year, month, day ->
                    val dateStr = String.format("%04d-%02d-%02d", year, month + 1, day)
                    onDateSelected(dateStr)
                },
                calendar.get(Calendar.YEAR),
                calendar.get(Calendar.MONTH),
                calendar.get(Calendar.DAY_OF_MONTH)
            )
            datePicker.show()
        }

        editStartDate.setOnClickListener { showDatePicker { editStartDate.setText(it) } }
        editEndDate.setOnClickListener { showDatePicker { editEndDate.setText(it) } }

        btnSearch.setOnClickListener {
            val startDate = editStartDate.text.toString()
            val endDate = editEndDate.text.toString()

            if (startDate.isBlank() || endDate.isBlank()) {
                Toast.makeText(this, "請選擇起訖日期", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            isDataLoaded = false
            loadingProgress.visibility = View.VISIBLE
            recyclerView.visibility = View.GONE

            apiService.getFallVideos(userId, startDate, endDate, limit = 5)
                .enqueue(object : Callback<List<VideoEvent>> {
                    override fun onResponse(call: Call<List<VideoEvent>>, response: Response<List<VideoEvent>>) {
                        loadingProgress.visibility = View.GONE
                        recyclerView.visibility = View.VISIBLE
                        if (response.isSuccessful && response.body() != null) {
                            val events = response.body()!!
                                .map {
                                    it.copy(user_id = userId, isFavorite = false, event_type = "fall")
                                }.toMutableList()
                            Log.d("VideoList", "取得事件數量：${events.size}")
                            adapter.updateData(events)
                            isDataLoaded = true
                        } else {
                            Toast.makeText(this@VideoListActivity, "查詢失敗 (${response.code()})，請稍後再試或確認伺服器是否開啟", Toast.LENGTH_LONG).show()
                        }
                    }

                    override fun onFailure(call: Call<List<VideoEvent>>, t: Throwable) {
                        loadingProgress.visibility = View.GONE
                        recyclerView.visibility = View.VISIBLE
                        Toast.makeText(this@VideoListActivity, "連線失敗：${t.message}，請確認網路或伺服器狀態", Toast.LENGTH_LONG).show()
                    }
                })
        }

        val today = Calendar.getInstance()
        val todayStr = String.format("%04d-%02d-%02d",
            today.get(Calendar.YEAR),
            today.get(Calendar.MONTH) + 1,
            today.get(Calendar.DAY_OF_MONTH))
        editStartDate.setText(todayStr)
        editEndDate.setText(todayStr)
        btnSearch.performClick()
    }

    private fun addToFavorite(event: VideoEvent) {
        val data = FavoriteRequest(event.user_id, event.video_filename, event.event_type)
        apiService.addFavorite(data).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    Toast.makeText(this@VideoListActivity, "已加入收藏", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@VideoListActivity, "加入收藏失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@VideoListActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun removeFromFavorite(event: VideoEvent) {
        val data = FavoriteRequest(event.user_id, event.video_filename, event.event_type)
        apiService.removeFavorite(data).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    Toast.makeText(this@VideoListActivity, "已取消收藏", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@VideoListActivity, "取消收藏失敗", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@VideoListActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
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
