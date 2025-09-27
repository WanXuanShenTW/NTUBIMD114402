package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.util.UnstableApi
import com.example.myapplication.model.LoginRequest
import com.example.myapplication.model.LoginResponse // = User 物件
import com.example.myapplication.model.LoginApiResponse // 新增的包裝回傳
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import androidx.core.widget.addTextChangedListener


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

        passwordInput.addTextChangedListener {
            clearError(passwordInput, passwordError)
        }
        phoneInput.addTextChangedListener {
            clearError(phoneInput, phoneError)
        }

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
            val phone = phoneInput.text.toString().trim()
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

            if (password.isBlank()) {
                showPasswordError("請輸入密碼")
                hasError = true
            } else if (!isValidPassword(password)) {
                showPasswordError("密碼格式不符（至少8碼，含大小寫）")
                hasError = true
            }

            if (hasError) return@setOnClickListener

            val req = LoginRequest(phone, password)

            RetrofitClient.apiService.loginUser(req)
                .enqueue(object : Callback<LoginApiResponse> {
                    override fun onResponse(
                        call: Call<LoginApiResponse>,
                        response: Response<LoginApiResponse>
                    ) {
                        // ① HTTP 層錯誤：解析 errorBody 的訊息
                        if (!response.isSuccessful) {
                            val code = response.code()
                            val msg = parseErrorMessage(response)  // 會抓 JSON 的 message/detail 或原始字串

                            when {
                                // 401 常用於密碼錯；有些後端也會回 400
                                code == 401 || msg.contains("密碼", ignoreCase = true) -> {
                                    showPasswordError("密碼錯誤")
                                }
                                // 404 常見「使用者不存在」；也可能回 400 搭配訊息
                                code == 404 || msg.contains("不存在", ignoreCase = true) || msg.contains("使用者", ignoreCase = true) || msg.contains("電話", ignoreCase = true) -> {
                                    showPhoneError("電話號碼不存在")
                                }
                                else -> {
                                    Toast.makeText(this@LoginActivity, if (msg.isNotBlank()) msg else "登入失敗（HTTP $code）", Toast.LENGTH_SHORT).show()
                                }
                            }
                            return
                        }

                        // ② 成功的 HTTP，但 success=false（你的新 API 格式）
                        val body = response.body()
                        if (body == null) {
                            Toast.makeText(this@LoginActivity, "登入失敗：伺服器回應為空", Toast.LENGTH_SHORT).show()
                            return
                        }
                        if (!body.success) {
                            val msg = body.message
                            when {
                                msg.contains("密碼", ignoreCase = true) -> showPasswordError("密碼錯誤")
                                msg.contains("不存在", ignoreCase = true) || msg.contains("使用者", ignoreCase = true) || msg.contains("電話", ignoreCase = true) -> showPhoneError("電話號碼不存在")
                                else -> Toast.makeText(this@LoginActivity, if (msg.isNotBlank()) msg else "登入失敗", Toast.LENGTH_SHORT).show()
                            }
                            return
                        }

                        // ③ 成功：照你原本的邏輯
                        val u: LoginResponse? = body.data?.user
                        if (u?.userId == null || u.userId <= 0) {
                            Toast.makeText(this@LoginActivity, "登入成功但缺少使用者資料", Toast.LENGTH_SHORT).show()
                            return
                        }

                        with(getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit()) {
                            putInt("user_id", u.userId)
                            putInt("role_id", u.roleId ?: -1)
                            putString("name", u.name ?: "")
                            putString("phone", u.phone ?: (phoneInput.text?.toString() ?: ""))
                            apply()
                        }

                        Toast.makeText(this@LoginActivity, "登入成功！", Toast.LENGTH_SHORT).show()
                        startActivity(Intent(this@LoginActivity, DashboardActivity::class.java))
                        finish()
                    }

                    override fun onFailure(call: Call<LoginApiResponse>, t: Throwable) {
                        Toast.makeText(
                            this@LoginActivity,
                            "登入失敗：${t.message ?: "連線錯誤"}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                })
        }
    }

    // —— 驗證/錯誤顯示工具 —— //

    private fun isValidPassword(p: String): Boolean {
        // 至少 8 碼、同時含大小寫（可含數字/符號）
        val hasUpper = p.any { it.isUpperCase() }
        val hasLower = p.any { it.isLowerCase() }
        return p.length >= 8 && hasUpper && hasLower
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

    private fun showPasswordError(msg: String = "密碼錯誤") {
        setError(passwordInput, passwordError, msg)
    }

    private fun showPhoneError(msg: String = "電話號碼不存在") {
        setError(phoneInput, phoneError, msg)
    }

    private fun parseErrorMessage(response: Response<*>): String {
        return try {
            val raw = response.errorBody()?.string()?.trim().orEmpty()
            if (raw.isBlank()) return ""
            try {
                val obj = org.json.JSONObject(raw)
                when {
                    obj.has("message") -> obj.optString("message")
                    obj.has("detail") -> {
                        val detail = obj.get("detail")
                        if (detail is String) detail
                        else detail.toString()
                    }
                    else -> raw
                }
            } catch (_: Exception) {
                raw // 不是 JSON 就原樣回傳
            }
        } catch (_: Exception) {
            ""
        }
    }
}
