package com.example.myapplication

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Path

class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var playerView: PlayerView
    private var player: ExoPlayer? = null
    private lateinit var tvDate: TextView
    private lateinit var backButton: ImageView
    private lateinit var headerOverlay: View
    private var isOverlayVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        playerView = findViewById(R.id.playerView)
        tvDate = findViewById(R.id.tvDate)
        backButton = findViewById(R.id.backButton)
        headerOverlay = findViewById(R.id.headerOverlay)

        playerView.setControllerVisibilityListener(PlayerView.ControllerVisibilityListener { visibilityState ->
            headerOverlay.animate().cancel()
            if (visibilityState == View.VISIBLE) {
                // 👉 直接出現，不加動畫
                headerOverlay.visibility = View.VISIBLE
                headerOverlay.alpha = 1f
            } else {
                // 👉 淡出動畫，動畫結束後設為 GONE
                headerOverlay.animate().alpha(0f).setDuration(200).withEndAction {
                    headerOverlay.visibility = View.GONE
                    headerOverlay.alpha = 1f // 重設透明度為 1，避免下次顯示時仍然是透明
                }.start()
            }
        })

        backButton.setOnClickListener {
            finish()
        }

        val videoUrl = intent.getStringExtra("video_url")
        val videoFilename = intent.getStringExtra("video_filename")

        Log.d("VideoPlayer", "收到的 video_url: $videoUrl")
        Log.d("VideoPlayer", "收到的 video_filename: $videoFilename")

        if (videoUrl.isNullOrEmpty()) {
            Toast.makeText(this, "無影片網址，無法播放", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        player = ExoPlayer.Builder(this).build().also { exoPlayer ->
            playerView.player = exoPlayer
            val mediaItem = MediaItem.fromUri(Uri.parse(videoUrl))
            exoPlayer.setMediaItem(mediaItem)

            exoPlayer.addListener(object : Player.Listener {
                override fun onPlayerError(error: PlaybackException) {
                    Toast.makeText(
                        this@VideoPlayerActivity,
                        "播放失敗：${error.errorCodeName}",
                        Toast.LENGTH_LONG
                    ).show()
                    Log.e("VideoPlayer", "播放錯誤：${error.errorCodeName}", error)
                }
            })

            exoPlayer.prepare()
            exoPlayer.playWhenReady = true
        }

        tvDate.text = "讀取中..."

        if (!videoFilename.isNullOrEmpty()) {
            val retrofit = Retrofit.Builder()
                .baseUrl("http://172.20.10.3:5000/")
                .addConverterFactory(GsonConverterFactory.create())
                .build()

            val api = retrofit.create(ApiService::class.java)

            api.getVideoDate(videoFilename).enqueue(object : Callback<VideoDateResponse> {
                override fun onResponse(
                    call: Call<VideoDateResponse>,
                    response: Response<VideoDateResponse>
                ) {
                    if (response.isSuccessful) {
                        tvDate.text = response.body()?.date ?: "未知日期"
                    } else {
                        tvDate.text = "讀取失敗"
                    }
                }

                override fun onFailure(call: Call<VideoDateResponse>, t: Throwable) {
                    tvDate.text = "錯誤：" + t.message
                }
            })
        } else {
            tvDate.text = "未提供檔名"
        }
    }

    override fun onStop() {
        super.onStop()
        player?.release()
        player = null
    }
}

// Retrofit 接口
interface ApiService {
    @GET("video-date/{video_filename}")
    fun getVideoDate(@Path("video_filename") filename: String): Call<VideoDateResponse>
}

// 回傳格式資料模型
data class VideoDateResponse(
    val video_filename: String,
    val date: String
)
