package com.example.myapplication

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.FavoriteAdapter
import com.example.myapplication.model.FavoriteRequest
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.network.ApiService
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import android.content.Context
import android.content.Intent
import com.example.myapplication.ApiConfig
import com.example.myapplication.network.RetrofitClient

@androidx.media3.common.util.UnstableApi
class FavoriteActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var loadingProgress: ProgressBar
    private lateinit var apiService: ApiService
    private lateinit var adapter: FavoriteAdapter
    private var userId: Int = -1  // 改為動態讀取

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_favorite)

        recyclerView = findViewById(R.id.recyclerFavorite)
        recyclerView.layoutManager = LinearLayoutManager(this)

        loadingProgress = findViewById(R.id.loadingProgress)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener {
            Log.d(TAG_FAV, "click: back to main") // ★ log
            finish()
        }

        val sharedPref = getSharedPreferences("smartcare_pref", Context.MODE_PRIVATE)
        userId = sharedPref.getInt("user_id", -1)
        val elderId = sharedPref.getString("elder_id", "") ?: ""

        Log.d(TAG_FAV, "onCreate | from prefs userId=$userId elderId=$elderId") // ★ log
        logPrefs("onCreate")

        apiService = RetrofitClient.apiService

        if (userId == -1) {
            Log.w(TAG_FAV, "onCreate | userId == -1 → go Login") // ★ log
            Toast.makeText(this, "尚未登入，請重新登入", Toast.LENGTH_LONG).show()
            startActivity(Intent(this, LoginActivity::class.java))
            finish()
            return
        }

        if (elderId.isBlank()) {
            Log.w(TAG_FAV, "onCreate | elderId blank → finish") // ★ log
            Toast.makeText(this, "請先選擇被照護者", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        Log.d(TAG_FAV, "onCreate → fetchFavorites(userId=$userId)") // ★ log
        fetchFavorites(userId)
    }

    override fun onResume() {
        super.onResume()
        val uid = getSharedPreferences("smartcare_pref", Context.MODE_PRIVATE)
            .getInt("user_id", -1)
        logPrefs("onResume") // ★ log
        if (uid > 0) {
            Log.d(TAG_FAV, "onResume → fetchFavorites(userId=$uid)") // ★ log
            fetchFavorites(uid)
        } else {
            Log.w(TAG_FAV, "onResume | uid <= 0, skip fetch") // ★ log
        }
    }

    private fun fetchFavorites(userId: Int) {
        loadingProgress.visibility = View.VISIBLE
        Log.d(TAG_FAV, "fetchFavorites() start | userId=$userId, type=fall")

        // 讀目前選的 elder_id
        val selectedElderId = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
            .getString("elder_id", null)?.toIntOrNull()

        apiService.getFavorites(userId, "fall").enqueue(object : Callback<List<VideoEvent>> {
            override fun onResponse(
                call: Call<List<VideoEvent>>,
                response: Response<List<VideoEvent>>
            ) {
                loadingProgress.visibility = View.GONE

                val raw = response.body().orEmpty()

                val selectedElderId = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
                    .getString("elder_id", null)?.toIntOrNull()

                val filtered = if (selectedElderId != null) {
                    raw.filter { it.user_id == selectedElderId }   // ← 這裡用 user_id 過濾
                } else raw

                Log.d(TAG_FAV, "getFavorites HTTP=${response.code()} size=${raw.size} filtered=${filtered.size} selectedElder=$selectedElderId")

                val favorites = filtered.map {
                    it.copy(
                        video_type = it.video_type ?: "fall",
                        in_watchlist = true,
                        isFavorite = true
                    )
                }.toMutableList()

                Log.d(TAG_FAV, "adapter set, items=${favorites.size}") // ★ log

                adapter = FavoriteAdapter(
                    favoriteList = favorites,
                    onRemoveClick = { video, _, done ->
                        Log.d(TAG_FAV, "remove click | record_id=${video.record_id}, type=${video.video_type}, curFav=${video.isFavorite}") // ★ log
                        val request = FavoriteRequest(
                            user_id = userId,
                            record_id = video.record_id,
                            video_type = video.video_type ?: "fall"
                        )
                        Log.d(TAG_FAV, "removeFavorite → $request")
                        apiService.removeFavorite(request).enqueue(object : Callback<Map<String, String>> {
                            override fun onResponse(
                                call: Call<Map<String, String>>,
                                response: Response<Map<String, String>>
                            ) {
                                Log.d(TAG_FAV, "removeFavorite HTTP=${response.code()} body=${response.body()}") // ★ log
                                if (response.isSuccessful) {
                                    Toast.makeText(this@FavoriteActivity, "已取消收藏", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(this@FavoriteActivity, "取消失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                                done()
                            }

                            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                                Log.e(TAG_FAV, "removeFavorite FAIL: ${t.message}", t) // ★ log
                                Toast.makeText(this@FavoriteActivity, "取消連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                                done()
                            }
                        })
                    },
                    onAddClick = { video, _, done ->
                        Log.d(TAG_FAV, "add click | record_id=${video.record_id}, type=${video.video_type}, curFav=${video.isFavorite}") // ★ log
                        val request = FavoriteRequest(
                            user_id = userId,
                            record_id = video.record_id,
                            video_type = video.video_type ?: "fall"
                        )
                        Log.d(TAG_FAV, "addFavorite → $request") // ★ log
                        apiService.addFavorite(request).enqueue(object : Callback<Map<String, String>> {
                            override fun onResponse(
                                call: Call<Map<String, String>>,
                                response: Response<Map<String, String>>
                            ) {
                                Log.d(TAG_FAV, "addFavorite HTTP=${response.code()} body=${response.body()}") // ★ log
                                if (response.isSuccessful) {
                                    Toast.makeText(this@FavoriteActivity, "已加入收藏", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(this@FavoriteActivity, "加入失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                                done()
                            }

                            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                                Log.e(TAG_FAV, "addFavorite FAIL: ${t.message}", t) // ★ log
                                Toast.makeText(this@FavoriteActivity, "加入連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                                done()
                            }
                        })
                    }
                )

                recyclerView.adapter = adapter
            }

            override fun onFailure(call: Call<List<VideoEvent>>, t: Throwable) {
                loadingProgress.visibility = View.GONE
                Log.e(TAG_FAV, "getFavorites FAIL: ${t.message}", t) // ★ log
                Log.e("收藏debug", "取得收藏失敗：${t.message}")
                Toast.makeText(this@FavoriteActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()

                adapter = FavoriteAdapter(
                    favoriteList = mutableListOf(),
                    onRemoveClick = { _, _, _ -> },
                    onAddClick = { _, _, _ -> }
                )
                recyclerView.adapter = adapter
            }
        })
    }

    private fun logPrefs(where: String) {
        val app = getSharedPreferences("app", MODE_PRIVATE)
        val smart = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        Log.d(
            TAG_FAV,
            "$where | app.user_id=${app.getInt("user_id", -1)}, app.contact_id=${app.getString("contact_id", null)} ; " +
                    "smart.user_id=${smart.getInt("user_id", -1)}, smart.elder_id=${smart.getString("elder_id", null)}"
        )
    }

    companion object {
        private const val TAG_FAV = "FavoritePage"
    }
}
