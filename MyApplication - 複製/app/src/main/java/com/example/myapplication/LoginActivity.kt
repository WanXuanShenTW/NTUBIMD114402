package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class LoginActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        val usernameInput = findViewById<EditText>(R.id.usernameInput)
        val phoneInput = findViewById<EditText>(R.id.phoneInput)
        val passwordInput = findViewById<EditText>(R.id.passwordInput)

        val forgotPasswordButton = findViewById<Button>(R.id.forgotPasswordButton)
        val registerButton = findViewById<Button>(R.id.registerButton)
        val loginButton = findViewById<Button>(R.id.loginButton)

        // ✅ 忘記密碼按鈕：導向驗證畫面
        forgotPasswordButton.setOnClickListener {
            val intent = Intent(this, VerifyPhoneActivity::class.java)
            startActivity(intent)
        }

        // 帳號註冊按鈕
        registerButton.setOnClickListener {
            Toast.makeText(this, "導向註冊頁面", Toast.LENGTH_SHORT).show()
        }

        //登入按鈕：直接跳轉主畫面（不驗證）
        loginButton.setOnClickListener {
            // 寫死固定使用者資料（可根據需要傳 user_id）
            val fakeUsername = "testuser"
            val fakePhone = "0912345678"
            val fakePassword = "123456"
            val fakeUserId = 1

            //選擇要跳的畫面，這裡是 DashboardActivity，你也可以換成 VideoListActivity
            val intent = Intent(this, DashboardActivity::class.java)
            intent.putExtra("user_id", fakeUserId)  // 需要的話可以傳給下一頁用
            startActivity(intent)
            finish() // 不讓使用者按返回鍵回到登入頁
        }
    }
}
