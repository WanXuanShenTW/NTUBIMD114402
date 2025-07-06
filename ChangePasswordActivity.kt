package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.example.myapplication.model.UpdateUserRequest
import com.example.myapplication.model.UpdateUserResponse
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class ChangePasswordActivity : AppCompatActivity() {

    private lateinit var editOldPassword: EditText
    private lateinit var editNewPassword: EditText
    private lateinit var editConfirmPassword: EditText
    private lateinit var btnSubmitChange: Button
    private lateinit var backButton: ImageView

    private lateinit var oldPasswordError: TextView
    private lateinit var newPasswordError: TextView
    private lateinit var confirmPasswordError: TextView

    private var phone: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_change_password)

        editOldPassword = findViewById(R.id.editOldPassword)
        editNewPassword = findViewById(R.id.editNewPassword)
        editConfirmPassword = findViewById(R.id.editConfirmPassword)
        btnSubmitChange = findViewById(R.id.btnSubmitChange)
        backButton = findViewById(R.id.backButton)

        oldPasswordError = findViewById(R.id.oldPasswordError)
        newPasswordError = findViewById(R.id.newPasswordError)
        confirmPasswordError = findViewById(R.id.confirmPasswordError)

        backButton.setOnClickListener {
            finish()
        }

        val sharedPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        phone = sharedPref.getString("phone", "") ?: ""

        btnSubmitChange.setOnClickListener {
            val oldPassword = editOldPassword.text.toString()
            val newPassword = editNewPassword.text.toString()
            val confirmPassword = editConfirmPassword.text.toString()

            // 清除錯誤
            clearError(editOldPassword, oldPasswordError)
            clearError(editNewPassword, newPasswordError)
            clearError(editConfirmPassword, confirmPasswordError)

            var hasError = false

            if (oldPassword.isEmpty()) {
                setError(editOldPassword, oldPasswordError, "請輸入舊密碼")
                hasError = true
            }

            if (newPassword.isEmpty()) {
                setError(editNewPassword, newPasswordError, "請輸入新密碼")
                hasError = true
            } else if (!isValidPassword(newPassword)) {
                setError(editNewPassword, newPasswordError, "密碼需至少8碼，含大寫與小寫字母")
                hasError = true
            }

            if (confirmPassword.isEmpty()) {
                setError(editConfirmPassword, confirmPasswordError, "請再次輸入新密碼")
                hasError = true
            } else if (newPassword != confirmPassword) {
                setError(editConfirmPassword, confirmPasswordError, "兩次輸入的新密碼不一致")
                hasError = true
            }

            if (hasError) return@setOnClickListener

            val updateRequest = UpdateUserRequest(
                phone = phone,
                name = null,
                oldPassword = oldPassword, // 假設後端已支援
                password = newPassword
            )

            RetrofitClient.apiService.updateUser(updateRequest)
                .enqueue(object : Callback<UpdateUserResponse> {
                    override fun onResponse(call: Call<UpdateUserResponse>, response: Response<UpdateUserResponse>) {
                        if (response.isSuccessful && response.body() != null) {
                            Toast.makeText(this@ChangePasswordActivity, "密碼修改成功，請重新登入", Toast.LENGTH_SHORT).show()
                            startActivity(Intent(this@ChangePasswordActivity, LoginActivity::class.java))
                            finish()
                        } else if (response.code() == 403) {
                            setError(editOldPassword, oldPasswordError, "舊密碼不正確")
                        } else {
                            Toast.makeText(this@ChangePasswordActivity, "修改失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    }

                    override fun onFailure(call: Call<UpdateUserResponse>, t: Throwable) {
                        Toast.makeText(this@ChangePasswordActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }
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
