package com.example.myapplication

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.doOnLayout
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView

@androidx.media3.common.util.UnstableApi
class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var playerView: PlayerView
    private var player: ExoPlayer? = null
    private lateinit var backButton: ImageView
    private lateinit var headerOverlay: View
    private lateinit var loadingOverlay: FrameLayout
    private var isHeaderVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        playerView = findViewById(R.id.playerView)
        backButton = findViewById(R.id.backButton)
        headerOverlay = findViewById(R.id.headerOverlay)
        loadingOverlay = findViewById(R.id.loadingOverlay)

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

        val videoUrl = "${ApiConfig.BASE_URL}fall_video_file?record_id=$recordId"
        Log.d("VideoPlayer", "播放 URL: $videoUrl")
        Log.d("VideoPlayer", "影片檔名: $videoFilename")

        player = ExoPlayer.Builder(this).build().also { exoPlayer ->
            playerView.player = exoPlayer
            playerView.controllerShowTimeoutMs = 2000
            playerView.showController()

            val mediaItem = MediaItem.fromUri(Uri.parse(videoUrl))
            exoPlayer.setMediaItem(mediaItem)

            exoPlayer.addListener(object : Player.Listener {
                override fun onPlaybackStateChanged(state: Int) {
                    when (state) {
                        Player.STATE_BUFFERING -> showLoading()
                        Player.STATE_READY, Player.STATE_ENDED -> hideLoading()
                    }
                }

                override fun onPlayerError(error: PlaybackException) {
                    hideLoading()
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
            showLoading()
        }

        // ✅ 用 onTouch 解決點擊無效問題
        playerView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                playerView.showController()

                if (!isHeaderVisible) {
                    headerOverlay.doOnLayout {
                        val height = it.height.takeIf { h -> h > 0 } ?: return@doOnLayout
                        isHeaderVisible = true
                        headerOverlay.animate().cancel()
                        headerOverlay.translationY = -height.toFloat()
                        headerOverlay.alpha = 0f
                        headerOverlay.visibility = View.VISIBLE
                        headerOverlay.animate()
                            .translationY(0f)
                            .alpha(1f)
                            .setDuration(300)
                            .start()
                    }
                }
            }
            false // 一定要回傳 false 才能保留 ExoPlayer 原本的點擊邏輯
        }

        playerView.setControllerVisibilityListener(
            PlayerView.ControllerVisibilityListener { visibility ->
                headerOverlay.doOnLayout {
                    val height = it.height.takeIf { h -> h > 0 } ?: return@doOnLayout
                    headerOverlay.animate().cancel()

                    if (visibility == View.VISIBLE && !isHeaderVisible) {
                        isHeaderVisible = true
                        headerOverlay.translationY = -height.toFloat()
                        headerOverlay.alpha = 0f
                        headerOverlay.visibility = View.VISIBLE
                        headerOverlay.animate()
                            .translationY(0f)
                            .alpha(1f)
                            .setDuration(300)
                            .start()
                    } else if (visibility != View.VISIBLE && isHeaderVisible) {
                        isHeaderVisible = false
                        headerOverlay.animate()
                            .translationY(-height.toFloat())
                            .alpha(0f)
                            .setDuration(300)
                            .withEndAction {
                                headerOverlay.visibility = View.INVISIBLE
                            }
                            .start()
                    }
                }
            }
        )
        // 強制讓 headerOverlay 顯示出來，確認有渲染成功
        headerOverlay.visibility = View.VISIBLE
        headerOverlay.alpha = 1f
        headerOverlay.translationY = 0f
        isHeaderVisible = true
    }

    private fun showLoading() {
        loadingOverlay.visibility = View.VISIBLE
    }

    private fun hideLoading() {
        loadingOverlay.visibility = View.GONE
    }

    override fun onStop() {
        super.onStop()
        player?.release()
        player = null
    }
}
