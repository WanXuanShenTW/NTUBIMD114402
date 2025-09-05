package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.util.UnstableApi
import com.example.myapplication.model.*
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

@UnstableApi
class EditProfileActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var spinnerRole: Spinner
    private lateinit var editLineId: EditText
    private lateinit var textPhone: TextView
    private lateinit var btnSave: Button
    private lateinit var btnCancel: ImageView   // 左上角返回圖示 or 取消編輯
    private lateinit var btnEdit: TextView      // 右上角「編輯」字樣
    private lateinit var btnChangePassword: Button
    private lateinit var btnDeleteAccount: Button
    private lateinit var loadingProgress: ProgressBar

    private var currentPhone: String = ""

    // 編輯狀態 & 原始值（用於取消編輯時還原）
    private var isEditing = false
    private var originalName = ""
    private var originalLineId = ""
    private var originalRolePos = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_edit_profile)

        Log.d("Lifecycle", "EditProfileActivity onCreate 呼叫中")

        editName = findViewById(R.id.editName)
        spinnerRole = findViewById(R.id.spinnerRole)
        editLineId = findViewById(R.id.editLineId)
        textPhone = findViewById(R.id.textPhone)
        btnSave = findViewById(R.id.btnSave)
        btnCancel = findViewById(R.id.btnCancel)
        btnEdit = findViewById(R.id.btnEdit)
        btnChangePassword = findViewById(R.id.btnChangePassword)
        btnDeleteAccount = findViewById(R.id.btnDeleteAccount)
        loadingProgress = findViewById(R.id.loadingProgress)

        val sharedPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        currentPhone = sharedPref.getString("phone", "") ?: ""
        Log.d("ProfileDebug", "sharedPref phone=$currentPhone")

        val currentName = sharedPref.getString("username", "")
        editName.setText(currentName)
        textPhone.text = currentPhone

        // 初始化角色下拉選單
        val roleAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, listOf("家屬", "醫護人員"))
        roleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerRole.adapter = roleAdapter
        spinnerRole.isEnabled = false

        // 初始狀態：非編輯
        toggleEditMode(false)
        snapshotCurrentFieldsAsOriginal()

        // 載入使用者資料
        if (currentPhone.isNotEmpty()) {
            loadingProgress.visibility = View.VISIBLE
            Log.d("ProfileDebug", "呼叫 getUser API")
            RetrofitClient.apiService.getUser(currentPhone)
                .enqueue(object : Callback<LoginResponse> {
                    override fun onResponse(call: Call<LoginResponse>, response: Response<LoginResponse>) {
                        loadingProgress.visibility = View.GONE
                        Log.d("ProfileDebug", "getUser HTTP=${response.code()} body=${response.body()} err=${response.errorBody()?.string()}")

                        val u = response.body() ?: run {
                            Toast.makeText(this@EditProfileActivity, "查無此用戶", Toast.LENGTH_SHORT).show()
                            return
                        }

                        editName.setText(u.name ?: "")
                        val ph = u.phone ?: ""
                        textPhone.text = ph
                        if (ph.isNotEmpty()) currentPhone = ph
                        editLineId.setText(u.lineId ?: "")

                        val rolePosition = when (u.roleId) {
                            2 -> 0 // 家屬
                            3 -> 1 // 醫護人員
                            else -> 0
                        }
                        spinnerRole.setSelection(rolePosition)

                        // 以 API 回傳為準，更新「原始值」
                        snapshotCurrentFieldsAsOriginal()
                    }

                    override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                        loadingProgress.visibility = View.GONE
                        Log.d("ProfileDebug", "getUser onFailure ${t.message}")
                        Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
        }

        // ===== 點擊事件 =====

        // 右上角「編輯」
        btnEdit.setOnClickListener {
            isEditing = true
            toggleEditMode(true)
        }

        // 左上角返回圖示：若在編輯中→取消編輯並還原；否則→直接關閉頁面
        btnCancel.setOnClickListener {
            if (isEditing) {
                restoreOriginalFields()
                isEditing = false
                toggleEditMode(false)
            } else {
                finish()
            }
        }

        btnSave.setOnClickListener {
            val newName = editName.text.toString().trim()
            val newLineId = editLineId.text.toString().trim()
            val newRoleId = when (spinnerRole.selectedItemPosition) {
                0 -> 2 // 家屬
                1 -> 3 // 醫護人員
                else -> 2
            }

            if (newName.isEmpty()) {
                Toast.makeText(this, "姓名不可為空", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            loadingProgress.visibility = View.VISIBLE
            btnSave.isEnabled = false

            val req = UpdateUserRequest(
                phone = currentPhone,
                name  = newName,
                lineId = newLineId,
                roleId = newRoleId
            )

            RetrofitClient.apiService.updateUser(req)
                .enqueue(object : Callback<UpdateUserResponse> {
                    override fun onResponse(
                        call: Call<UpdateUserResponse>,
                        response: Response<UpdateUserResponse>
                    ) {
                        loadingProgress.visibility = View.GONE
                        btnSave.isEnabled = true

                        if (response.isSuccessful) {
                            Toast.makeText(this@EditProfileActivity, "已更新", Toast.LENGTH_SHORT).show()

                            // 同步本地快取
                            getSharedPreferences("smartcare_pref", MODE_PRIVATE)
                                .edit()
                                .putString("username", newName)
                                .putString("line_id", newLineId)
                                .apply()

                            // 關閉編輯模式
                            snapshotCurrentFieldsAsOriginal()
                            isEditing = false
                            toggleEditMode(false)

                            // 重新拉最新資料（以伺服器為準）
                            fetchAndFill()
                        } else {
                            Toast.makeText(
                                this@EditProfileActivity,
                                "更新失敗：${response.code()}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }

                    override fun onFailure(call: Call<UpdateUserResponse>, t: Throwable) {
                        loadingProgress.visibility = View.GONE
                        btnSave.isEnabled = true
                        Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                    }
                })
            }

        btnChangePassword.setOnClickListener {
            Log.d("ClickDebug", "點擊了 修改密碼")
            startActivity(Intent(this, ChangePasswordActivity::class.java))
        }

        // 刪除帳號
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

    private fun fetchAndFill() {
        if (currentPhone.isEmpty()) return
        loadingProgress.visibility = View.VISIBLE
        RetrofitClient.apiService.getUser(currentPhone)
            .enqueue(object : Callback<LoginResponse> {
                override fun onResponse(call: Call<LoginResponse>, response: Response<LoginResponse>) {
                    loadingProgress.visibility = View.GONE
                    val u = response.body() ?: run {
                        Toast.makeText(this@EditProfileActivity, "查無此用戶", Toast.LENGTH_SHORT).show()
                        return
                    }
                    editName.setText(u.name ?: "")
                    val ph = u.phone ?: ""
                    textPhone.text = ph
                    if (ph.isNotEmpty()) currentPhone = ph
                    editLineId.setText(u.lineId ?: "")

                    val rolePosition = when (u.roleId) {
                        2 -> 0 // 家屬
                        3 -> 1 // 醫護人員
                        else -> 0
                    }
                    spinnerRole.setSelection(rolePosition)

                    snapshotCurrentFieldsAsOriginal()
                }

                override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                    loadingProgress.visibility = View.GONE
                    Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    // 依編輯狀態切換 UI
    private fun toggleEditMode(isEditable: Boolean) {
        editName.isEnabled = isEditable
        spinnerRole.isEnabled = isEditable
        editLineId.isEnabled = isEditable

        btnSave.visibility = if (isEditable) View.VISIBLE else View.GONE
        btnSave.isEnabled = isEditable

        btnEdit.visibility = if (isEditable) View.GONE else View.VISIBLE

        btnChangePassword.visibility = if (isEditable) View.VISIBLE else View.GONE
        btnChangePassword.isEnabled = isEditable
        btnDeleteAccount.visibility = if (isEditable) View.VISIBLE else View.GONE
        btnDeleteAccount.isEnabled = isEditable
    }

    // 記錄目前欄位作為「原始值」
    private fun snapshotCurrentFieldsAsOriginal() {
        originalName = editName.text?.toString() ?: ""
        originalLineId = editLineId.text?.toString() ?: ""
        originalRolePos = spinnerRole.selectedItemPosition
    }

    // 還原到「原始值」
    private fun restoreOriginalFields() {
        editName.setText(originalName)
        editLineId.setText(originalLineId)
        spinnerRole.setSelection(originalRolePos)
    }

    // 實體返回鍵：編輯中先取消；否則直接返回
    override fun onBackPressed() {
        if (isEditing) {
            restoreOriginalFields()
            isEditing = false
            toggleEditMode(false)
        } else {
            super.onBackPressed()
        }
    }
}
