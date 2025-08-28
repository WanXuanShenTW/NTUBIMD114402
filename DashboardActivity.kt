package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.util.UnstableApi
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.ElderAdapter
import com.example.myapplication.model.DeleteEmergencyContactRequest
import com.example.myapplication.model.EmergencyContact
import com.example.myapplication.model.EmergencyContactRequest
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import android.util.Log
import org.json.JSONObject
import com.example.myapplication.model.LoginResponse

@UnstableApi
class DashboardActivity : AppCompatActivity() {

    data class ElderItem(
        val contactId: Int,   // ← 資料庫 contact_id
        val phone: String,    // 顯示用/備用
        val display: String   // 列表顯示名稱
    )

    private var userId: Int = -1
    private lateinit var txtSelectedElder: TextView
    private lateinit var btnAddElder: ImageButton
    private val apiService = RetrofitClient.apiService
    private var elderList = mutableListOf<ElderItem>()
    private lateinit var caregiverPhone: String

    override fun onCreate(savedInstanceState: Bundle?) {
        run {
            val appSp = getSharedPreferences("app", MODE_PRIVATE)
            val old = appSp.getString("contact_id", null)
            if (!old.isNullOrEmpty() && old.any { !it.isDigit() }) {
                appSp.edit().remove("contact_id").apply()
            }
        }

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dashboard)

        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        userId = smartPref.getInt("user_id", -1)
        if (userId == -1) {
            startActivity(Intent(this, LoginActivity::class.java))
            finish()
            return
        }

        val appPref = getSharedPreferences("app", MODE_PRIVATE)
        if (userId > 0) {
            appPref.edit().putInt("user_id", userId).apply()
        } else {
            android.util.Log.w("N8N_PREF", "smart.user_id 無效：$userId，先不寫入 app.user_id")
        }

        logPrefs("onCreate-afterWrite")

        txtSelectedElder = findViewById(R.id.txtSelectedElder)
        btnAddElder = findViewById(R.id.btnAddElder)
        btnAddElder.visibility = View.GONE

        // 若有上次選擇的 elder_name 就顯示（只影響顯示，不動 contact_id）
        smartPref.getString("elder_name", null)?.let { lastName ->
            if (lastName.isNotBlank()) txtSelectedElder.text = lastName
        }

        // 點擊可選擇被照護者
        txtSelectedElder.setOnClickListener {
            if (elderList.isEmpty()) {
                Toast.makeText(this, "尚未載入被照護者", Toast.LENGTH_SHORT).show()
            } else {
                showSelectElderDialog()
            }
        }

        // 載入被照護者（在 loadContacts 內你會自動帶第一筆 contact_id）
        loadContacts()

        // ---- 其它導覽按鈕 ----
        findViewById<ImageButton>(R.id.navcollect).setOnClickListener {
            startActivity(Intent(this, FavoriteActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btnLivingRoom).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }
        findViewById<ImageButton>(R.id.navChat).setOnClickListener {
            startActivity(Intent(this, VoiceResultActivity::class.java))
        }
        findViewById<ImageButton>(R.id.navHistory).setOnClickListener {
            startActivity(Intent(this, VideoListActivity::class.java))
        }
        findViewById<ImageButton>(R.id.navDocument).setOnClickListener {
            startActivity(Intent(this, ReportActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btnmenu).setOnClickListener {
            startActivity(Intent(this, EditProfileActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btnLogout).setOnClickListener {
            // 清 smartcare 與 app 兩份偏好
            getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit().clear().apply()
            getSharedPreferences("app", MODE_PRIVATE).edit().clear().apply()
            // 回登入並清掉返回棧
            startActivity(Intent(this, LoginActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
        }
    }

    private fun logPrefs(where: String) {
        val a = getSharedPreferences("app", MODE_PRIVATE)
        val s = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        android.util.Log.d(
            "N8N_PREF",
            "$where | app.user_id=${a.getInt("user_id", -1)}, app.contact_id=${a.getString("contact_id", null)} ; " +
                    "smart.phone=${s.getString("phone", null)}, smart.elder_id=${s.getString("elder_id", null)}, " +
                    "smart.elder_name=${s.getString("elder_name", null)}"
        )
    }

    private fun showSelectElderDialog() {
        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val dialogView = layoutInflater.inflate(R.layout.dialog_select_elder, null)
        val dialogRecycler = dialogView.findViewById<RecyclerView>(R.id.dialogRecyclerElder)
        val btnDialogAddElder = dialogView.findViewById<ImageButton>(R.id.btnDialogAddElder)

        dialogRecycler.layoutManager = LinearLayoutManager(this)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialogRecycler.adapter = ElderAdapter(
            // 顯示：contactId -> display
            elderList.map { it.contactId.toString() to it.display },

            onItemClick = { elder: Pair<String, String> ->
                val contactIdStr = elder.first
                val display = elder.second

                // 顯示
                txtSelectedElder.text = display

                // 記在 smartcare_pref（給 UI 用）
                smartPref.edit()
                    .putString("elder_id", contactIdStr)
                    .putString("elder_name", display)
                    .apply()

                // 記在 app（給 N8nSender 用）
                getSharedPreferences("app", MODE_PRIVATE).edit()
                    .putInt("user_id", userId)              // 保險再寫一次
                    .putString("contact_id", contactIdStr)  // 關鍵
                    .apply()

                Log.d("SelectElder", "saved user_id=$userId, contact_id=$contactIdStr")
                Toast.makeText(this, "已選擇 $display", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            },

            onItemLongClick = l@{ elder: Pair<String, String> ->
                val cid = elder.first.toIntOrNull() ?: return@l
                val picked = elderList.firstOrNull { it.contactId == cid } ?: return@l

                val caregiverPhone = smartPref.getString("phone", "") ?: ""
                if (caregiverPhone.isBlank()) {
                    Toast.makeText(this, "無法取得登入者手機，無法刪除", Toast.LENGTH_SHORT).show()
                    return@l
                }

                AlertDialog.Builder(this)
                    .setTitle("刪除被照護者")
                    .setMessage("確定要刪除 ${picked.display} 嗎？")
                    .setPositiveButton("刪除") { _, _ ->
                        deleteContact(userPhone = caregiverPhone, contactPhone = picked.phone)
                    }
                    .setNegativeButton("取消", null)
                    .show()
            }
        )

        btnDialogAddElder.setOnClickListener {
            dialog.dismiss()
            showAddElderDialog()
        }

        dialog.show()
    }

    private fun showAddElderDialog() {
        val inputView = layoutInflater.inflate(R.layout.dialog_add_elder, null)
        val editPhone = inputView.findViewById<EditText>(R.id.editContactPhone)

        AlertDialog.Builder(this)
            .setTitle("新增被照護者")
            .setView(inputView)
            .setPositiveButton("新增") { _, _ ->
                val phone = editPhone.text.toString()
                if (!phone.matches(Regex("^09\\d{8}$"))) {
                    Toast.makeText(this, "手機格式錯誤，需為09開頭共10碼", Toast.LENGTH_SHORT).show()
                } else {
                    addNewElder(phone)
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    fun saveIds(context: Context, userId: Int, contactId: String) {
        context.getSharedPreferences("app", Context.MODE_PRIVATE)
            .edit()
            .putInt("user_id", userId)
            .putString("contact_id", contactId)
            .apply()
        Log.d("N8N_PREF", "saved user_id=$userId, contact_id=$contactId")
    }

    private fun addNewElder(phone: String) {
        val sharedPref = getSharedPreferences("smartcare_pref", Context.MODE_PRIVATE)
        val userPhone = sharedPref.getString("phone", "") ?: ""

        // 先查這個電話對應的角色
        RetrofitClient.apiService.getUser(phone)
            .enqueue(object : Callback<LoginResponse> {
                override fun onResponse(call: Call<LoginResponse>, response: Response<LoginResponse>) {
                    if (!response.isSuccessful) {
                        Toast.makeText(this@DashboardActivity, "查詢失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val body = response.body() ?: run {
                        Toast.makeText(this@DashboardActivity, "查無此用戶", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val roleId = body.roleId ?: -1   // ← 這裡用 roleId（駝峰）
                    if (roleId != 1) {
                        Toast.makeText(this@DashboardActivity, "此用戶非被照護者，無法新增", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val request = EmergencyContactRequest(
                        user_phone = userPhone,
                        contact_phone = phone,
                        priority = 1,
                        relationship = ""
                    )

                    RetrofitClient.apiService.addEmergencyContact(request)
                        .enqueue(object : Callback<Map<String, String>> {
                            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                                if (response.isSuccessful) {
                                    Toast.makeText(this@DashboardActivity, "新增成功", Toast.LENGTH_SHORT).show()
                                    loadContacts()
                                } else {
                                    val raw = response.errorBody()?.string()
                                    val msg = try {
                                        val json = JSONObject(raw ?: "")
                                        json.optString("error", "新增失敗：${response.code()}")
                                    } catch (_: Exception) {
                                        "新增失敗：${response.code()}"
                                    }
                                    Toast.makeText(this@DashboardActivity, msg, Toast.LENGTH_SHORT).show()
                                }
                            }

                            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                                Toast.makeText(this@DashboardActivity, "新增失敗：${t.message}", Toast.LENGTH_SHORT).show()
                            }
                        })
                }

                override fun onFailure(call: Call<LoginResponse>, t: Throwable) {
                    Toast.makeText(this@DashboardActivity, "查詢失敗：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun loadContacts() {
        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val appPref   = getSharedPreferences("app", MODE_PRIVATE)
        val userPhone = smartPref.getString("phone", "") ?: ""

        RetrofitClient.apiService.getEmergencyContacts(userPhone, 1)
            .enqueue(object : Callback<List<EmergencyContact>> {
                override fun onResponse(
                    call: Call<List<EmergencyContact>>,
                    response: Response<List<EmergencyContact>>
                ) {
                    try {
                        if (response.isSuccessful && response.body() != null) {
                            val contacts = response.body()!!
                            elderList.clear()

                            if (contacts.isEmpty()) {
                                Toast.makeText(
                                    this@DashboardActivity,
                                    "目前沒有設定被照護者",
                                    Toast.LENGTH_SHORT
                                ).show()
                                // 沒資料時可清掉 contact_id（避免 N8nSender 誤送）
                                appPref.edit().remove("contact_id").apply()
                                smartPref.edit().remove("elder_id").remove("elder_name").apply()
                                txtSelectedElder.text = ""
                                logPrefs("loadContacts-empty")
                                return
                            }

                            elderList.addAll(
                                contacts.mapNotNull { c ->
                                    val cid   = c.userId ?: return@mapNotNull null
                                    val phone = c.phone  ?: return@mapNotNull null
                                    val display = buildString {
                                        if (!c.name.isNullOrBlank()) append(c.name)
                                        if (!c.roleName.isNullOrBlank()) append(" (${c.roleName})")
                                        if (!c.relationship.isNullOrBlank()) append(" (${c.relationship})")
                                        if (isNotEmpty()) append(" - ")
                                        append(phone)
                                    }
                                    ElderItem(contactId = cid, phone = phone, display = display)
                                }
                            )

                            // 取第一筆自動帶入（僅在 app.contact_id 不存在或不是純數字時）
                            val appPref = getSharedPreferences("app", MODE_PRIVATE)
                            val existingCid = appPref.getString("contact_id", null)
                            if (elderList.isNotEmpty() && (existingCid.isNullOrBlank() || !existingCid.all(Char::isDigit))) {
                                val first = elderList.first() // ElderItem(contactId=被照護者user_id, ...)
                                appPref.edit().putString("contact_id", first.contactId.toString()).apply()

                                // 同步 UI 用的 smartcare_pref（之後你手動點選也會覆蓋）
                                getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit()
                                    .putString("elder_id", first.contactId.toString())
                                    .putString("elder_name", first.display)
                                    .apply()

                                txtSelectedElder.text = first.display
                                Log.d("AutoPick", "Auto selected elder contact_id=${first.contactId} (${first.display})")
                                logPrefs("afterAutoPick") // 方便你在 Logcat 看
                            }

                            val savedCidStr = smartPref.getString("elder_id", null)
                            val preferred = savedCidStr?.toIntOrNull()
                                ?.let { cid -> elderList.firstOrNull { it.contactId == cid } }

                            // 沒有就選第一筆
                            val chosen = preferred ?: elderList.first()

                            // 寫回兩份偏好：smartcare_pref（UI 用）與 app（N8nSender 用）
                            smartPref.edit()
                                .putString("elder_id", chosen.contactId.toString())
                                .putString("elder_name", chosen.display)
                                .apply()

                            appPref.edit()
                                .putInt("user_id", userId)
                                .putString("contact_id", chosen.contactId.toString())
                                .apply()

                            // 更新畫面
                            txtSelectedElder.text = chosen.display

                            logPrefs("loadContacts-autoSelect")
                            // ---------- 自動選擇完成 ----------
                        } else {
                            val raw = response.errorBody()?.string()
                            Log.e("DashboardDebug", "載入失敗 code=${response.code()}, body=$raw")
                            Toast.makeText(
                                this@DashboardActivity,
                                "載入失敗：${response.code()}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } catch (e: Exception) {
                        Log.e("DashboardDebug", "解析錯誤: ${e.message}", e)
                        Toast.makeText(
                            this@DashboardActivity,
                            "解析錯誤：${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onFailure(call: Call<List<EmergencyContact>>, t: Throwable) {
                    Log.e("DashboardDebug", "API 請求失敗: ${t.message}", t)
                    Toast.makeText(
                        this@DashboardActivity,
                        "連線錯誤：${t.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            })
    }

    private fun deleteContact(userPhone: String, contactPhone: String) {
        Log.d("DeleteDebug", "準備刪除，送出的參數 user_phone=$userPhone, contact_phone=$contactPhone")

        val request = DeleteEmergencyContactRequest(
            user_phone = userPhone,
            contact_phone = contactPhone
        )

        Log.d("DeleteDebug", "發送 JSON = $request")

        RetrofitClient.apiService.deleteEmergencyContact(request)
            .enqueue(object : Callback<Map<String, String>> {
                override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                    try {
                        val raw = response.errorBody()?.string()
                        Log.e("DeleteDebug", "刪除回應 code=${response.code()}, body=$raw")

                        if (response.isSuccessful) {
                            val removed = elderList.removeAll { it.phone == contactPhone }
                            Log.d("DeleteDebug", "本地已移除: $removed")
                            Toast.makeText(this@DashboardActivity, "刪除成功", Toast.LENGTH_SHORT).show()
                            loadContacts()
                        } else {
                            val msg = try {
                                val json = JSONObject(raw ?: "")
                                json.optString("error")
                            } catch (e: Exception) {
                                null
                            }

                            Log.e("DeleteDebug", "JSON 解析後的 error=$msg")

                            when {
                                msg?.contains("找不到") == true || raw?.contains("找不到") == true -> {
                                    Toast.makeText(this@DashboardActivity, "無此照護關係", Toast.LENGTH_SHORT).show()
                                }
                                else -> {
                                    Toast.makeText(this@DashboardActivity, "刪除失敗 ${response.code()}", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("DeleteDebug", "刪除解析失敗: ${e.message}", e)
                        Toast.makeText(this@DashboardActivity, "刪除失敗（解析錯誤）", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                    Log.e("DeleteDebug", "刪除請求失敗: ${t.message}", t)
                    Toast.makeText(this@DashboardActivity, "刪除錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }
}
