package com.example.myapplication

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView

class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var playerView: PlayerView
    private var player: ExoPlayer? = null
    private lateinit var backButton: ImageView
    private lateinit var headerOverlay: View
    private var isOverlayVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        playerView = findViewById(R.id.playerView)
        backButton = findViewById(R.id.backButton)
        headerOverlay = findViewById(R.id.headerOverlay)

        playerView.setControllerVisibilityListener(PlayerView.ControllerVisibilityListener { visibilityState ->
            headerOverlay.animate().cancel()
            if (visibilityState == View.VISIBLE) {
                headerOverlay.visibility = View.VISIBLE
                headerOverlay.alpha = 1f
            } else {
                headerOverlay.animate().alpha(0f).setDuration(200).withEndAction {
                    headerOverlay.visibility = View.GONE
                    headerOverlay.alpha = 1f
                }.start()
            }
        })

        backButton.setOnClickListener {
            finish()
        }

        val recordId = intent.getIntExtra("record_id", -1)
        val videoFilename = intent.getStringExtra("video_filename")

        if (recordId == -1) {
            Toast.makeText(this, "未提供影片 ID，無法播放", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val videoUrl = "https://c8fd-60-250-79-110.ngrok-free.app/fall_video_file?record_id=$recordId"

        Log.d("VideoPlayer", "播放 URL: $videoUrl")
        Log.d("VideoPlayer", "影片檔名: $videoFilename")

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
    }

    override fun onStop() {
        super.onStop()
        player?.release()
        player = null
    }
}
