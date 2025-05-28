package com.example.myapplication

import android.os.Bundle
import android.util.Log
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.FavoriteAdapter
import com.example.myapplication.model.FavoriteResponse
import com.example.myapplication.model.FavoriteVideo
import com.example.myapplication.network.ApiService
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory

class FavoriteActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var apiService: ApiService
    private var userId: Int = 1  // 👉 之後可改成動態傳入的

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_favorite)

        recyclerView = findViewById(R.id.recyclerFavorite)
        recyclerView.layoutManager = LinearLayoutManager(this)

        // 返回主畫面
        findViewById<LinearLayout>(R.id.btnBackToMain).setOnClickListener {
            finish()
        }

        val retrofit = Retrofit.Builder()
            .baseUrl("http://172.20.10.3:5000/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        apiService = retrofit.create(ApiService::class.java)

        fetchFavorites()
    }

    private fun fetchFavorites() {
        apiService.getFavorites(userId).enqueue(object : Callback<FavoriteResponse> {
            override fun onResponse(
                call: Call<FavoriteResponse>,
                response: Response<FavoriteResponse>
            ) {
                if (response.isSuccessful && response.body() != null) {
                    val favorites: List<FavoriteVideo> = response.body()!!.favorites
                    val adapter = FavoriteAdapter(favorites)
                    recyclerView.adapter = adapter
                } else {
                    Toast.makeText(this@FavoriteActivity, "讀取失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<FavoriteResponse>, t: Throwable) {
                Log.e("收藏debug", "取得收藏失敗：${t.message}")
                Toast.makeText(this@FavoriteActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }
}
