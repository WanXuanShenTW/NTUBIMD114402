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
import com.example.myapplication.model.FavoriteVideo
import com.example.myapplication.network.ApiService
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import com.example.myapplication.ApiConfig

@androidx.media3.common.util.UnstableApi
class FavoriteActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var loadingProgress: ProgressBar
    private lateinit var apiService: ApiService
    private lateinit var adapter: FavoriteAdapter
    private var userId: Int = 3  // 👉 可改為動態傳入

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_favorite)

        recyclerView = findViewById(R.id.recyclerFavorite)
        recyclerView.layoutManager = LinearLayoutManager(this)

        loadingProgress = findViewById(R.id.loadingProgress)

        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener {
            finish()
        }

        val retrofit = Retrofit.Builder()
            .baseUrl("${ApiConfig.BASE_URL}")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        apiService = retrofit.create(ApiService::class.java)

        fetchFavorites()
    }

    private fun fetchFavorites() {
        loadingProgress.visibility = View.VISIBLE

        apiService.getFavorites(userId).enqueue(object : Callback<List<FavoriteVideo>> {
            override fun onResponse(
                call: Call<List<FavoriteVideo>>,
                response: Response<List<FavoriteVideo>>
            ) {
                loadingProgress.visibility = View.GONE

                val favorites = response.body()?.toMutableList() ?: mutableListOf()
                favorites.forEach { it.is_favorited = true }

                adapter = FavoriteAdapter(
                    favoriteList = favorites,
                    onRemoveClick = { video, _, done ->
                        val request = FavoriteRequest(
                            user_id = userId,
                            record_id = video.record_id,
                            video_type = video.video_type
                        )
                        apiService.removeFavorite(request).enqueue(object : Callback<Map<String, String>> {
                            override fun onResponse(
                                call: Call<Map<String, String>>,
                                response: Response<Map<String, String>>
                            ) {
                                if (response.isSuccessful) {
                                    Toast.makeText(this@FavoriteActivity, "已取消收藏", Toast.LENGTH_SHORT).show()
                                    done()
                                } else {
                                    Toast.makeText(this@FavoriteActivity, "取消失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                            }

                            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                                Toast.makeText(this@FavoriteActivity, "取消連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                            }
                        })
                    },
                    onAddClick = { video, _, done ->
                        val request = FavoriteRequest(
                            user_id = userId,
                            record_id = video.record_id,
                            video_type = video.video_type
                        )
                        apiService.addFavorite(request).enqueue(object : Callback<Map<String, String>> {
                            override fun onResponse(
                                call: Call<Map<String, String>>,
                                response: Response<Map<String, String>>
                            ) {
                                if (response.isSuccessful) {
                                    Toast.makeText(this@FavoriteActivity, "已加入收藏", Toast.LENGTH_SHORT).show()
                                    done()
                                } else {
                                    Toast.makeText(this@FavoriteActivity, "加入失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                            }

                            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                                Toast.makeText(this@FavoriteActivity, "加入連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                            }
                        })
                    }
                )

                recyclerView.adapter = adapter
            }

            override fun onFailure(call: Call<List<FavoriteVideo>>, t: Throwable) {
                loadingProgress.visibility = View.GONE

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
}
