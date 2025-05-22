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
import com.example.myapplication.network.ApiService
import com.example.myapplication.adapter.VideoEventAdapter
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import java.util.*

class VideoListActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: VideoEventAdapter

    private lateinit var editUserId: EditText
    private lateinit var editStartDate: EditText
    private lateinit var editEndDate: EditText
    private lateinit var btnSearch: Button

    private lateinit var apiService: ApiService
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

        val retrofit = Retrofit.Builder()
            .baseUrl("http://172.20.10.3:5000/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        apiService = retrofit.create(ApiService::class.java)

        adapter = VideoEventAdapter(
            listOf(),
            onItemClick = { event ->
                val videoUrl = "http://172.20.10.3:5000/videos/${event.event_type}/${event.video_filename}"
                val intent = Intent(this, VideoPlayerActivity::class.java).apply {
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
                    addToFavorite(event)
                } else {
                    removeFromFavorite(event)
                }
            }
        )

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        fun showDatePicker(onDateSelected: (String) -> Unit) {
            val calendar = Calendar.getInstance()
            val datePicker = DatePickerDialog(this,
                { _, year, month, day ->
                    val dateStr = String.format(Locale.getDefault(), "%04d-%02d-%02d", year, month + 1, day)
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
            userId = editUserId.text.toString().toIntOrNull() ?: -1
            val startDate = editStartDate.text.toString()
            val endDate = editEndDate.text.toString()

            if (userId == -1 || startDate.isBlank() || endDate.isBlank()) {
                Toast.makeText(this, "請輸入使用者ID與起訖日期", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            isDataLoaded = false

            apiService.getVideoEvents(userId, startDate, endDate)
                .enqueue(object : Callback<List<VideoEvent>> {
                    override fun onResponse(call: Call<List<VideoEvent>>, response: Response<List<VideoEvent>>) {
                        if (response.isSuccessful && response.body() != null) {
                            val eventsWithUserId = response.body()!!.map {
                                it.copy(user_id = userId)
                            }
                            adapter.updateData(eventsWithUserId)
                            isDataLoaded = true
                        } else {
                            Toast.makeText(this@VideoListActivity, "查詢失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    }

                    override fun onFailure(call: Call<List<VideoEvent>>, t: Throwable) {
                        Toast.makeText(this@VideoListActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }
    }

    private fun addToFavorite(event: VideoEvent) {
        val data = FavoriteRequest(
            user_id = event.user_id,
            video_filename = event.video_filename,
            video_type = event.event_type
        )
        Log.d("FavoriteDebug", "送出收藏請求：user_id=${data.user_id}, filename=${data.video_filename}, type=${data.video_type}")


        apiService.addFavorite(data).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    Toast.makeText(this@VideoListActivity, "已加入收藏", Toast.LENGTH_SHORT).show()
                } else if (response.code() == 409) {
                    Toast.makeText(this@VideoListActivity, "此影片已在收藏清單中", Toast.LENGTH_SHORT).show()
                } else {
                    val errorBody = response.errorBody()?.string()
                    Log.e("FavoriteDebug", "收藏失敗：${response.code()}，錯誤內容：$errorBody")
                    Toast.makeText(this@VideoListActivity, "收藏失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@VideoListActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun removeFromFavorite(event: VideoEvent) {
        val data = FavoriteRequest(
            user_id = event.user_id,
            video_filename = event.video_filename,
            video_type = event.event_type
        )

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
