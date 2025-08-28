package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.util.UnstableApi
import com.example.myapplication.model.LoginRequest
import com.example.myapplication.model.LoginResponse
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

@UnstableApi
class LoginActivity : AppCompatActivity() {

    private lateinit var phoneInput: EditText
    private lateinit var passwordInput: EditText
    private lateinit var phoneError: TextView
    private lateinit var passwordError: TextView
    private lateinit var forgotPasswordButton: Button
    private lateinit var registerButton: Button
    private lateinit var loginButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        phoneInput = findViewById(R.id.phoneInput)
        passwordInput = findViewById(R.id.passwordInput)
        phoneError = findViewById(R.id.phoneError)
        passwordError = findViewById(R.id.passwordError)

        forgotPasswordButton = findViewById(R.id.forgotPasswordButton)
        registerButton = findViewById(R.id.registerButton)
        loginButton = findViewById(R.id.loginButton)

        forgotPasswordButton.setOnClickListener {
            startActivity(Intent(this, VerifyPhoneActivity::class.java))
        }

        registerButton.setOnClickListener {
            startActivity(Intent(this, RegisterActivity::class.java))
        }

        loginButton.setOnClickListener {
            val phone = phoneInput.text.toString()
            val password = passwordInput.text.toString()

            clearError(phoneInput, phoneError)
            clearError(passwordInput, passwordError)

            var hasError = false

            if (phone.isEmpty()) {
                setError(phoneInput, phoneError, "請輸入手機號碼")
                hasError = true
            } else if (!phone.matches(Regex("^09\\d{8}$"))) {
                setError(phoneInput, phoneError, "手機號碼格式錯誤（需為09開頭共10碼）")
                hasError = true
            }

            if (password.isEmpty()) {
                setError(passwordInput, passwordError, "請輸入密碼")
                hasError = true
            } else if (!password.matches(Regex("^(?=.*[a-z])(?=.*[A-Z]).{8,}$"))) {
                setError(passwordInput, passwordError, "密碼需至少8碼，且包含大小寫字母")
                hasError = true
            }

            if (hasError) return@setOnClickListener

            val loginRequest = LoginRequest(phone, password)
            RetrofitClient.apiService.getUser(phone)
                .enqueue(object : Callback<LoginResponse> {
                    override fun onResponse(call: Call<LoginResponse>, r: Response<LoginResponse>) {
                        if (!r.isSuccessful) {
                            Toast.makeText(this@LoginActivity, "登入成功，但讀取使用者失敗：${r.code()}", Toast.LENGTH_SHORT).show()
                            return
                        }

                        // 先把 body 拿出來；為空就結束（之後 u 一定是非空）
                        val u = r.body() ?: run {
                            Toast.makeText(this@LoginActivity, "登入成功，但回應為空", Toast.LENGTH_SHORT).show()
                            return
                        }

                        val uid = u.userId ?: -1
                        if (uid <= 0) {
                            Toast.makeText(this@LoginActivity, "登入成功，但查無有效 user_id", Toast.LENGTH_SHORT).show()
                            return
                        }

                        with(getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit()) {
                            putInt("user_id", uid)
                            putInt("role_id", u.roleId ?: -1)
                            putString("name", u.name ?: "")
                            putString("phone", u.phone ?: phone)   // 以回應為主，沒有就用輸入的 phone
                            putString("line_id", u.lineId)
                            apply()
                        }

                        Toast.makeText(this@LoginActivity, "登入成功！", Toast.LENGTH_SHORT).show()
                        startActivity(Intent(this@LoginActivity, DashboardActivity::class.java))
                        finish()
                    }

                    override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                        Toast.makeText(this@LoginActivity, "登入成功，但讀取使用者失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }
    }

    private fun setError(editText: EditText, errorText: TextView, message: String) {
        editText.setBackgroundResource(R.drawable.edittext_error_background)
        errorText.text = message
        errorText.visibility = View.VISIBLE
    }

    private fun clearError(editText: EditText, errorText: TextView) {
        editText.setBackgroundResource(R.drawable.edittext_background)
        errorText.visibility = View.GONE
    }
}