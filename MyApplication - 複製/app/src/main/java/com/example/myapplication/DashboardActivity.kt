package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity

class DashboardActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dashboard)

        // 👉 收藏影片（點擊下載圖示）
        findViewById<ImageButton>(R.id.navDownload).setOnClickListener {
            val intent = Intent(this, FavoriteActivity::class.java)
            intent.putExtra("user_id", 1) //之後要改成動態id
            startActivity(intent)
        }

        findViewById<ImageButton>(R.id.btnLivingRoom).setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        // 👉 影片歷史紀錄
        findViewById<ImageButton>(R.id.navHistory).setOnClickListener {
            startActivity(Intent(this, VideoListActivity::class.java))
        }

        // 👉 登出按鈕
        findViewById<ImageButton>(R.id.btnLogout).setOnClickListener {
            finishAffinity()  // 關閉所有畫面，回到登入或退出 app
        }
    }
}
