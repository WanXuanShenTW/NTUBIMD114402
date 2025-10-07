package com.example.myapplication

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatButton
import androidx.media3.common.util.UnstableApi
import com.example.myapplication.model.*
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import com.example.myapplication.AppKeys
import androidx.appcompat.app.AlertDialog
import android.widget.TextView
import android.content.Context



@UnstableApi
class EditProfileActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var textPhone: TextView
    private lateinit var editAddress: EditText
    private lateinit var btnSave: Button
    private lateinit var btnCancel: ImageView
    private lateinit var btnEdit: TextView
    private lateinit var btnChangePassword: Button
    private lateinit var btnDeleteAccount: Button
    private lateinit var linkAddLine: TextView
    private lateinit var loadingProgress: ProgressBar
    private lateinit var spinnerProfileType: Spinner

    private var currentPhone: String = ""
    private var elderId: Int = -1

    // 編輯狀態 & 原始值（用於取消還原）
    private var isEditing = false
    private var originalName = ""
    private var originalAddress = ""

    companion object {
        private const val FB_APP_ID = "1149819690311439"
        private const val REDIRECT_URI =
            "https://09fc9ad21a04.ngrok-free.app/webhook/e807ba71-0dc1-48df-b393-c6c6f03e17c1"
        private const val FB_SCOPE = "public_profile,user_posts"
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_edit_profile)

        Log.d("Lifecycle", "EditProfileActivity onCreate 呼叫中")

        // --- UI 綁定 ---
        editName = findViewById(R.id.editName)
        textPhone = findViewById(R.id.textPhone)
        editAddress = findViewById(R.id.editAddress)   // 注意：XML 要改成這個 ID
        btnSave = findViewById(R.id.btnSave)
        btnCancel = findViewById(R.id.btnCancel)
        btnEdit = findViewById(R.id.btnEdit)
        btnChangePassword = findViewById(R.id.btnChangePassword)
        btnDeleteAccount = findViewById(R.id.btnDeleteAccount)
        loadingProgress = findViewById(R.id.loadingProgress)
        spinnerProfileType = findViewById(R.id.spinnerProfileType)

        linkAddLine = findViewById(R.id.linkAddLine)
        linkAddLine.paintFlags = linkAddLine.paintFlags or android.graphics.Paint.UNDERLINE_TEXT_FLAG
        linkAddLine.setOnClickListener { openLineOfficialAccount(AppKeys.LINE_OA_ID) }

        val sharedPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        currentPhone = sharedPref.getString("phone", "") ?: ""
        elderId = sharedPref.getString("elder_id", "-1")?.toIntOrNull() ?: -1

        // 用自訂的 item 樣式
        val profileAdapter = ArrayAdapter.createFromResource(
            this,
            R.array.profile_types,            // strings.xml 裡的 <string-array>
            R.layout.spinner_item             // 自訂顯示樣式（字大、間距調整）
        )
        profileAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item)
        spinnerProfileType.adapter = profileAdapter

        val caregiverPhone = sharedPref.getString("phone", "") ?: ""
        val appPref = getSharedPreferences("app", MODE_PRIVATE)
        val elderPhone = appPref.getString("elder_phone", "") ?: ""

        // 初始狀態：非編輯
        toggleEditMode(false)

        val linkAddFacebook = findViewById<TextView>(R.id.linkAddFacebook)
        linkAddFacebook.setOnClickListener {
            val sp = getSharedPreferences(AppKeys.SP, Context.MODE_PRIVATE)
            val caregiverUserId = sp.getInt(AppKeys.USER_ID, -1)
            if (caregiverUserId <= 0) {
                Toast.makeText(this, "尚未登入或找不到使用者ID", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val state = "$caregiverUserId"

            val oauthUri = Uri.parse("https://www.facebook.com/v23.0/dialog/oauth")
                .buildUpon()
                .appendQueryParameter("client_id", FB_APP_ID)
                .appendQueryParameter("redirect_uri", REDIRECT_URI)
                .appendQueryParameter("response_type", "code")
                .appendQueryParameter("scope", FB_SCOPE)
                .appendQueryParameter("state", state)
                .build()

            Log.d("FB_AUTH", "uid=$caregiverUserId")
            Log.d("FB_AUTH", "oauthUri=$oauthUri")

            startActivity(Intent(Intent.ACTION_VIEW, oauthUri))
        }

        spinnerProfileType.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                if (pos == 0) {
                    // 照護者 → 顯示功能按鈕
                    findViewById<View>(R.id.labelAddress).visibility = View.GONE
                    findViewById<View>(R.id.editAddress).visibility = View.GONE
                    findViewById<View>(R.id.linkAddFacebook).visibility = View.VISIBLE
                    btnChangePassword.visibility = View.VISIBLE
                    btnDeleteAccount.visibility = View.VISIBLE
                    linkAddLine.visibility = View.VISIBLE
                    fetchCaregiverProfile(caregiverPhone)
                } else {
                    // 被照護者 → 隱藏功能按鈕
                    findViewById<View>(R.id.linkAddFacebook).visibility = View.GONE
                    findViewById<View>(R.id.labelAddress).visibility = View.GONE
                    findViewById<View>(R.id.editAddress).visibility = View.GONE
                    btnChangePassword.visibility = View.GONE
                    btnDeleteAccount.visibility = View.GONE
                    linkAddLine.visibility = View.GONE
                    if (elderPhone.isNotEmpty()) {
                        fetchUserProfile(elderPhone)
                    } else {
                        Toast.makeText(this@EditProfileActivity, "尚未選擇被照護者", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // ===== 事件 =====
        btnEdit.setOnClickListener {
            if (!isEditing) {
                // 進入編輯
                isEditing = true
                toggleEditMode(true)
            } else {
                // 取消編輯：還原並退出
                restoreOriginalFields()
                isEditing = false
                toggleEditMode(false)
            }
        }

        btnDeleteAccount.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("刪除帳號")
                .setMessage("確定要刪除此帳號嗎？此動作無法復原。")
                .setPositiveButton("確定") { _, _ ->
                    deleteAccount(currentPhone)
                }
                .setNegativeButton("取消", null)
                .show()
        }

        btnCancel.setOnClickListener {
            if (isEditing) {
                restoreOriginalFields()
                isEditing = false
                toggleEditMode(false)
            } else {
                finish()
            }
        }

        btnSave.setOnTouchListener { _, ev ->
            android.util.Log.d("EditProfile", "[SAVE_TOUCH] action=${ev.action}")
            false // 不攔截，讓 onClick 照常跑
        }

        btnSave.setOnClickListener {
            android.util.Log.d("EditProfile", "[SAVE_CLICK] pressed; isEditing=$isEditing")

            if (!isEditing) {
                android.util.Log.w("EditProfile", "[SAVE_ABORT] not in edit mode")
                return@setOnClickListener
            }

            val newName = editName.text.toString().trim()
            if (newName.isEmpty()) {
                android.util.Log.w("EditProfile", "[SAVE_ABORT] empty name")
                return@setOnClickListener
            }

            val selectedType = spinnerProfileType.selectedItemPosition // 0=照護者, 1=被照護者
            val caregiverPhone = getSharedPreferences("smartcare_pref", MODE_PRIVATE).getString("phone","") ?: ""
            val elderPhone = getSharedPreferences("app", MODE_PRIVATE).getString("elder_phone","") ?: ""
            val targetPhone = if (selectedType == 0) caregiverPhone else elderPhone
            val address = if (selectedType == 1) editAddress.text.toString().trim().ifBlank { null } else null

            android.util.Log.d("EditProfile", "[SAVE_PRECHECK] type=$selectedType phone=$targetPhone address=$address")
            if (targetPhone.isBlank()) {
                val who = if (selectedType == 0) "照護者" else "被照護者"
                Toast.makeText(this, "$who 的電話不存在，請先載入資料", Toast.LENGTH_SHORT).show()
                android.util.Log.w("EditProfile", "[SAVE_ABORT] phone blank")
                return@setOnClickListener
            }

            updateUserProfile(targetPhone, newName, selectedType, address)
        }

////        btnSave.setOnClickListener {
////            Log.d("EditProfile", "[SAVE_CLICK] pressed; isEditing=$isEditing")
////            Toast.makeText(this, "準備送出…", Toast.LENGTH_SHORT).show()
////
////            val newName = editName.text.toString().trim()
////            if (newName.isEmpty()) {
////                Toast.makeText(this, "姓名不可為空", Toast.LENGTH_SHORT).show()
////                Log.w("EditProfile", "[SAVE_ABORT] empty name")
////                return@setOnClickListener
////            }
////
////            val selectedType = spinnerProfileType.selectedItemPosition // 0=照護者, 1=被照護者
////            val caregiverPhone = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
////                .getString("phone", "") ?: ""
////            val elderPhone = getSharedPreferences("app", MODE_PRIVATE)
////                .getString("elder_phone", "") ?: ""
////            val targetPhone = if (selectedType == 0) caregiverPhone else elderPhone
////            val address = if (selectedType == 1) editAddress.text.toString().trim().ifBlank { null } else null
////
////            Log.d("EditProfile", "[SAVE_PRECHECK] type=$selectedType phone=$targetPhone address=$address")
////
////            if (targetPhone.isBlank()) {
////                val who = if (selectedType == 0) "照護者" else "被照護者"
////                Toast.makeText(this, "$who 的電話不存在，請先載入資料", Toast.LENGTH_SHORT).show()
////                Log.w("EditProfile", "[SAVE_ABORT] phone blank")
////                return@setOnClickListener
////            }
//
//            updateUserProfile(targetPhone, newName, selectedType, address)
//        }

        btnChangePassword.setOnClickListener {
            startActivity(Intent(this, ChangePasswordActivity::class.java))
        }

        // ProgressBar 不要擋點擊
        loadingProgress.visibility = View.GONE
        loadingProgress.isClickable = false
        loadingProgress.isFocusable = false
    }

    private fun updateUserProfile(phone: String, name: String, type: Int, address: String?) {
        android.util.Log.d("EditProfile", "[API_CALL] phone=$phone name=$name type=$type address=$address")
        loadingProgress.visibility = View.VISIBLE
        btnSave.isEnabled = false

        val req = UpdateUserRequest(
            phone   = phone,
            name    = name,
            roleId  = if (type == 0) 2 else 1,
            address = address
        )
        android.util.Log.d("EditProfile", "[API_REQ] $req")

        RetrofitClient.apiService.updateUser(req)
            .enqueue(object : Callback<UpdateUserResponse> {
                override fun onResponse(call: Call<UpdateUserResponse>, response: Response<UpdateUserResponse>) {
                    loadingProgress.visibility = View.GONE
                    btnSave.isEnabled = true
                    if (response.isSuccessful) {
                        snapshotCurrentFieldsAsOriginal()
                        toggleEditMode(false)
                    } else {
                        Toast.makeText(this@EditProfileActivity, "更新失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }
                override fun onFailure(call: Call<UpdateUserResponse>, t: Throwable) {
                    loadingProgress.visibility = View.GONE
                    btnSave.isEnabled = true
                    Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun deleteAccount(phone: String) {
        if (phone.isEmpty()) {
            Toast.makeText(this, "無效的帳號", Toast.LENGTH_SHORT).show()
            return
        }

        loadingProgress.visibility = View.VISIBLE

        val req = DeleteUserRequest(phone)

        RetrofitClient.apiService.deleteUser(req)
            .enqueue(object : Callback<DeleteUserResponse> {
                override fun onResponse(
                    call: Call<DeleteUserResponse>,
                    response: Response<DeleteUserResponse>
                ) {
                    loadingProgress.visibility = View.GONE
                    if (response.isSuccessful) {
                        val res = response.body()
                        if (res?.message?.contains("成功") == true) {
                            Toast.makeText(this@EditProfileActivity, "帳號已刪除", Toast.LENGTH_SHORT).show()

                            // 清掉 SharedPreferences
                            getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit().clear().apply()
                            getSharedPreferences("app", MODE_PRIVATE).edit().clear().apply()

                            // 跳回登入頁
                            val intent = Intent(this@EditProfileActivity, LoginActivity::class.java)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                            startActivity(intent)
                            finish()
                        } else {
                            Toast.makeText(
                                this@EditProfileActivity,
                                res?.message ?: "刪除失敗",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } else {
                        Toast.makeText(this@EditProfileActivity, "刪除失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onFailure(call: Call<DeleteUserResponse>, t: Throwable) {
                    loadingProgress.visibility = View.GONE
                    Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun fetchCaregiverProfile(phone: String) {
        if (phone.isEmpty()) return
        loadingProgress.visibility = View.VISIBLE

        RetrofitClient.apiService.getUser(phone)
            .enqueue(object : Callback<ApiResponse<UserDto>> {
                override fun onResponse(
                    call: Call<ApiResponse<UserDto>>,
                    response: Response<ApiResponse<UserDto>>
                ) {
                    loadingProgress.visibility = View.GONE
                    if (!response.isSuccessful) {
                        Toast.makeText(this@EditProfileActivity, "查詢失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val body = response.body()
                    val u = body?.data
                    if (u == null) {
                        Toast.makeText(this@EditProfileActivity, body?.message ?: "無資料", Toast.LENGTH_SHORT).show()
                        return
                    }

                    editName.setText(u.name.orEmpty())
                    textPhone.text = u.phone.orEmpty()
                    currentPhone = u.phone.orEmpty()

                    findViewById<TextView>(R.id.textRole).text = when (u.roleId) {
                        1 -> "長者"
                        2 -> "家屬"
                        else -> "未知"
                    }

                    snapshotCurrentFieldsAsOriginal()
                }

                override fun onFailure(call: Call<ApiResponse<UserDto>>, t: Throwable) {
                    loadingProgress.visibility = View.GONE
                    Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun fetchUserProfile(phone: String) {
        if (phone.isEmpty()) return
        loadingProgress.visibility = View.VISIBLE

        RetrofitClient.apiService.getUser(phone)
            .enqueue(object : Callback<ApiResponse<UserDto>> {
                override fun onResponse(
                    call: Call<ApiResponse<UserDto>>,
                    response: Response<ApiResponse<UserDto>>
                ) {
                    loadingProgress.visibility = View.GONE
                    if (!response.isSuccessful) {
                        Toast.makeText(this@EditProfileActivity, "查詢失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val body = response.body()
                    val u = body?.data
                    if (u == null) {
                        Toast.makeText(this@EditProfileActivity, body?.message ?: "無資料", Toast.LENGTH_SHORT).show()
                        return
                    }

                    editName.setText(u.name.orEmpty())
                    textPhone.text = u.phone.orEmpty()
                    findViewById<TextView>(R.id.textRole).text = "長者"
                    editAddress.setText(u.address.orEmpty())

                    snapshotCurrentFieldsAsOriginal()
                }

                override fun onFailure(call: Call<ApiResponse<UserDto>>, t: Throwable) {
                    loadingProgress.visibility = View.GONE
                    Toast.makeText(this@EditProfileActivity, "連線失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun toggleEditMode(isEditable: Boolean) {
        isEditing = isEditable

        editName.isEnabled = isEditable
        editAddress.isEnabled = isEditable

        // 儲存：只有編輯時可見
        btnSave.visibility  = if (isEditable) View.VISIBLE else View.GONE
        btnSave.isEnabled   = isEditable
        btnSave.isClickable = isEditable

        // 右上角按鈕：永遠可見，但文字切換
        btnEdit.visibility = View.VISIBLE
        btnEdit.text = if (isEditable) "– 取消" else "+ 編輯"
    }

    private fun snapshotCurrentFieldsAsOriginal() {
        originalName = editName.text?.toString() ?: ""
        originalAddress = editAddress.text?.toString() ?: ""
    }

    private fun restoreOriginalFields() {
        editName.setText(originalName)
        editAddress.setText(originalAddress)
    }

    private fun openLineOfficialAccount(oaId: String) {
        val deepLink = Uri.parse("line://ti/p/@$oaId")
        try {
            startActivity(Intent(Intent.ACTION_VIEW, deepLink))
            return
        } catch (_: Exception) {}

        val webAddFriend = Uri.parse("https://line.me/R/ti/p/@$oaId")
        try {
            startActivity(Intent(Intent.ACTION_VIEW, webAddFriend))
        } catch (_: Exception) {
            val profilePage = Uri.parse("https://page.line.me/$oaId")
            try {
                startActivity(Intent(Intent.ACTION_VIEW, profilePage))
            } catch (_: Exception) {
                Toast.makeText(this, "無法開啟 LINE 或瀏覽器", Toast.LENGTH_SHORT).show()
            }
        }
    }

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
