package com.example.myapplication

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import android.content.Intent
import com.example.myapplication.model.RegisterRequest
import com.example.myapplication.model.RegisterResponse
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class RegisterActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var editPhone: EditText
    private lateinit var editPassword: EditText
    private lateinit var editConfirmPassword: EditText
    private lateinit var btnRegister: Button
    private lateinit var backButton: ImageView

    private lateinit var nameError: TextView
    private lateinit var phoneError: TextView
    private lateinit var passwordError: TextView
    private lateinit var confirmPasswordError: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_register)

        editName = findViewById(R.id.editName)
        editPhone = findViewById(R.id.editPhone)
        editPassword = findViewById(R.id.editPassword)
        editConfirmPassword = findViewById(R.id.editConfirmPassword)
        btnRegister = findViewById(R.id.btnRegister)
        backButton = findViewById(R.id.backButton)

        nameError = findViewById(R.id.nameError)
        phoneError = findViewById(R.id.phoneError)
        passwordError = findViewById(R.id.passwordError)
        confirmPasswordError = findViewById(R.id.confirmPasswordError)

        backButton.setOnClickListener {
            finish()
        }

        btnRegister.setOnClickListener {
            val name = editName.text.toString()
            val phone = editPhone.text.toString()
            val password = editPassword.text.toString()
            val confirmPassword = editConfirmPassword.text.toString()

            // 清除所有錯誤狀態
            clearError(editName, nameError)
            clearError(editPhone, phoneError)
            clearError(editPassword, passwordError)
            clearError(editConfirmPassword, confirmPasswordError)

            var hasError = false

            // 姓名檢查
            if (name.isEmpty()) {
                setError(editName, nameError, "姓名不能為空")
                hasError = true
            }

            // 手機檢查
            if (phone.isEmpty()) {
                setError(editPhone, phoneError, "手機號碼不能為空")
                hasError = true
            } else if (!isValidPhone(phone)) {
                setError(editPhone, phoneError, "手機號碼格式錯誤（需為09開頭共10碼）")
                hasError = true
            }

            // 密碼檢查
            if (password.isEmpty()) {
                setError(editPassword, passwordError, "請輸入密碼")
                hasError = true
            } else if (!isValidPassword(password)) {
                setError(editPassword, passwordError, "密碼需至少8碼，含大寫與小寫英文字母")
                hasError = true
            }

            // 確認密碼檢查
            if (confirmPassword.isEmpty()) {
                setError(editConfirmPassword, confirmPasswordError, "請再次輸入密碼")
                hasError = true
            } else if (password != confirmPassword) {
                setError(editConfirmPassword, confirmPasswordError, "兩次密碼不一致")
                hasError = true
            }

            if (hasError) return@setOnClickListener

            // ✅ 呼叫後端 API 註冊
            val registerRequest = RegisterRequest(
                name = name,
                phone = phone,
                password = password,
                role_id = 2
            )

            RetrofitClient.apiService.registerUser(registerRequest)
                .enqueue(object : Callback<RegisterResponse> {
                    override fun onResponse(
                        call: Call<RegisterResponse>,
                        response: Response<RegisterResponse>
                    ) {
                        if (response.isSuccessful && response.body() != null) {
                            Toast.makeText(this@RegisterActivity, "註冊成功！請重新登入", Toast.LENGTH_SHORT).show()
                            startActivity(Intent(this@RegisterActivity, LoginActivity::class.java))
                            finish()
                        } else if (response.code() == 409) {
                            // 改用紅色提示顯示
                            setError(editPhone, phoneError, "手機號碼已被註冊")
                        } else {
                            val errorMsg = response.errorBody()?.string() ?: "未知錯誤"
                            Toast.makeText(this@RegisterActivity, "註冊失敗：$errorMsg", Toast.LENGTH_LONG).show()
                        }
                    }

                    override fun onFailure(call: Call<RegisterResponse>, t: Throwable) {
                        Toast.makeText(this@RegisterActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }
    }

    private fun isValidPhone(phone: String): Boolean {
        val regex = Regex("^09\\d{8}$")
        return phone.matches(regex)
    }

    private fun isValidPassword(password: String): Boolean {
        val regex = Regex("^(?=.*[a-z])(?=.*[A-Z]).{8,}$")
        return password.matches(regex)
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
