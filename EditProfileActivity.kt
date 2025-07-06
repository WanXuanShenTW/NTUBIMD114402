package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.myapplication.model.*
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class EditProfileActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var textPhone: TextView
    private lateinit var textUserId: TextView
    private lateinit var btnSave: Button
    private lateinit var btnCancel: ImageView
    private lateinit var btnEdit: TextView
    private lateinit var btnChangePassword: Button
    private lateinit var btnDeleteAccount: Button
    private lateinit var loadingProgress: ProgressBar

    private var userId: Int = -1
    private var currentPhone: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_edit_profile)

        Log.d("Lifecycle", "EditProfileActivity onCreate 呼叫中")

        editName = findViewById(R.id.editName)
        textPhone = findViewById(R.id.textPhone)
        textUserId = findViewById(R.id.textUserId)
        btnSave = findViewById(R.id.btnSave)
        btnCancel = findViewById(R.id.btnCancel)
        btnEdit = findViewById(R.id.btnEdit)
        btnChangePassword = findViewById(R.id.btnChangePassword)
        btnDeleteAccount = findViewById(R.id.btnDeleteAccount)
        loadingProgress = findViewById(R.id.loadingProgress)

        val sharedPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        userId = sharedPref.getInt("user_id", -1)
        currentPhone = sharedPref.getString("phone", "") ?: ""
        Log.d("ProfileDebug", "sharedPref user_id=$userId phone=$currentPhone")

        val currentName = sharedPref.getString("username", "")
        editName.setText(currentName)
        textPhone.text = currentPhone
        textUserId.text = userId.toString()

        toggleEditMode(false)

        if (currentPhone.isNotEmpty()) {
            loadingProgress.visibility = View.VISIBLE
            Log.d("ProfileDebug", "呼叫 getUser API")
            RetrofitClient.apiService.getUser(currentPhone)
                .enqueue(object : Callback<LoginResponse> {
                    override fun onResponse(call: Call<LoginResponse>, response: Response<LoginResponse>) {
                        loadingProgress.visibility = View.GONE
                        Log.d("ProfileDebug", "getUser HTTP=${response.code()} body=${response.body()} error=${response.errorBody()?.string()}")
                        if (response.isSuccessful && response.body() != null) {
                            val userInfo = response.body()!!
                            editName.setText(userInfo.name)
                            textPhone.text = userInfo.phone
                            currentPhone = userInfo.phone

                            userId = userInfo.user_id
                            textUserId.text = userId.toString()
                        } else {
                            Toast.makeText(this@EditProfileActivity, "載入失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    }
                    override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                        loadingProgress.visibility = View.GONE
                        Log.d("ProfileDebug", "getUser onFailure ${t.message}")
                        Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }

        btnEdit.setOnClickListener {
            Log.d("ClickDebug", "點擊了 +編輯")
            toggleEditMode(true)
            btnSave.visibility = View.VISIBLE
            btnSave.isEnabled = true
            Toast.makeText(this, "已啟用編輯模式", Toast.LENGTH_SHORT).show()
        }

        btnCancel.setOnClickListener {
            Log.d("ClickDebug", "點擊了 返回")
            finish()
        }

        btnSave.setOnClickListener {
            Log.d("ClickDebug", "點擊了 儲存")
            val newName = editName.text.toString()
            if (newName.isEmpty()) {
                Toast.makeText(this, "請輸入完整資訊", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val updateRequest = UpdateUserRequest(
                phone = currentPhone,
                name = newName,
                password = null
            )
            Log.d("UpdateDebug", "準備呼叫 updateUser PATCH JSON=$updateRequest")
            btnSave.isEnabled = false
            loadingProgress.visibility = View.VISIBLE

            RetrofitClient.apiService.updateUser(updateRequest)
                .enqueue(object : Callback<UpdateUserResponse> {
                    override fun onResponse(call: Call<UpdateUserResponse>, response: Response<UpdateUserResponse>) {
                        loadingProgress.visibility = View.GONE
                        btnSave.isEnabled = true
                        Log.d("UpdateDebug", "updateUser HTTP=${response.code()} body=${response.body()} error=${response.errorBody()?.string()}")
                        if (response.isSuccessful && response.body() != null) {
                            Toast.makeText(this@EditProfileActivity, "修改成功！", Toast.LENGTH_SHORT).show()
                            sharedPref.edit()
                                .putString("username", newName)
                                .putString("phone", currentPhone)
                                .apply()
                            toggleEditMode(false)
                        } else {
                            Toast.makeText(this@EditProfileActivity, "修改失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    }

                    override fun onFailure(call: Call<UpdateUserResponse>, t: Throwable) {
                        loadingProgress.visibility = View.GONE
                        btnSave.isEnabled = true
                        Log.d("UpdateDebug", "updateUser onFailure ${t.message}")
                        Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }

        btnChangePassword.setOnClickListener {
            Log.d("ClickDebug", "點擊了 修改密碼")
            startActivity(Intent(this, ChangePasswordActivity::class.java))
        }

        btnDeleteAccount.setOnClickListener {
            Log.d("ClickDebug", "點擊了 刪除帳號")
            AlertDialog.Builder(this)
                .setTitle("確認刪除帳號")
                .setMessage("此動作無法復原，確定要刪除嗎？")
                .setPositiveButton("確定") { _, _ ->
                    Log.d("DeleteDebug", "使用者按下確認刪除，準備呼叫API")
                    val deleteRequest = DeleteUserRequest(currentPhone)
                    loadingProgress.visibility = View.VISIBLE
                    RetrofitClient.apiService.deleteUser(deleteRequest)
                        .enqueue(object : Callback<DeleteUserResponse> {
                            override fun onResponse(call: Call<DeleteUserResponse>, response: Response<DeleteUserResponse>) {
                                loadingProgress.visibility = View.GONE
                                Log.d("DeleteDebug", "deleteUser HTTP=${response.code()} body=${response.body()} error=${response.errorBody()?.string()}")
                                if (response.isSuccessful) {
                                    Toast.makeText(this@EditProfileActivity, "帳號刪除成功", Toast.LENGTH_SHORT).show()
                                    sharedPref.edit().clear().apply()
                                    val intent = Intent(this@EditProfileActivity, LoginActivity::class.java)
                                    intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                                    startActivity(intent)
                                } else {
                                    Toast.makeText(this@EditProfileActivity, "刪除失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                            }

                            override fun onFailure(call: Call<DeleteUserResponse>, t: Throwable) {
                                loadingProgress.visibility = View.GONE
                                Log.d("DeleteDebug", "deleteUser onFailure ${t.message}")
                                Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                            }
                        })
                }
                .setNegativeButton("取消", null)
                .show()
        }
    }

    private fun toggleEditMode(isEditable: Boolean) {
        editName.isEnabled = isEditable
        if (isEditable) {
            btnSave.visibility = View.VISIBLE
            btnChangePassword.visibility = View.VISIBLE
            btnDeleteAccount.visibility = View.VISIBLE
        } else {
            btnSave.visibility = View.GONE
            btnChangePassword.visibility = View.GONE
            btnDeleteAccount.visibility = View.GONE
        }
    }
}
