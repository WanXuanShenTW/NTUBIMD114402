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
        val contactId: Int,   // ← 資料庫 contact_id（或被照護者 user_id）
        val phone: String,
        val display: String
    )

    private var userId: Int = -1
    private lateinit var txtSelectedElder: TextView
    private lateinit var btnAddElder: ImageButton
    private var elderList = mutableListOf<ElderItem>()
    private var selectElderDlg: AlertDialog? = null
    private var selectElderRecycler: RecyclerView? = null


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
            Log.w("N8N_PREF", "smart.user_id 無效：$userId，先不寫入 app.user_id")
        }

        logPrefs("onCreate-afterWrite")

        txtSelectedElder = findViewById(R.id.txtSelectedElder)
        btnAddElder = findViewById(R.id.btnAddElder)
        btnAddElder.visibility = View.GONE

        smartPref.getString("elder_name", null)?.let { lastName ->
            if (lastName.isNotBlank()) txtSelectedElder.text = lastName
        }

        txtSelectedElder.setOnClickListener {
            if (elderList.isEmpty()) {
                Toast.makeText(this, "尚未載入被照護者", Toast.LENGTH_SHORT).show()
            } else {
                showSelectElderDialog()
            }
        }

        loadContacts()

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
            getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit().clear().apply()
            getSharedPreferences("app", MODE_PRIVATE).edit().clear().apply()
            startActivity(Intent(this, LoginActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
        }
    }

    private fun logPrefs(where: String) {
        val a = getSharedPreferences("app", MODE_PRIVATE)
        val s = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        Log.d(
            "N8N_PREF",
            "$where | app.user_id=${a.getInt("user_id", -1)}, app.contact_id=${a.getString("contact_id", null)} ; " +
                    "smart.phone=${s.getString("phone", null)}, smart.elder_id=${s.getString("elder_id", null)}, " +
                    "smart.elder_name=${s.getString("elder_name", null)}"
        )
    }

    private fun showSelectElderDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_select_elder, null)
        val dialogRecycler = dialogView.findViewById<RecyclerView>(R.id.dialogRecyclerElder)
        val btnDialogAddElder = dialogView.findViewById<ImageButton>(R.id.btnDialogAddElder)

        dialogRecycler.layoutManager = LinearLayoutManager(this)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        // 重要：把參考存起來，之後刪除成功可即時刷新列表
        selectElderDlg = dialog
        selectElderRecycler = dialogRecycler
        rebuildElderDialogList()  // 依目前的 elderList 綁定 Adapter

        btnDialogAddElder.setOnClickListener {
            dialog.dismiss()
            showAddElderDialog()
        }

        dialog.show()
    }

    private fun rebuildElderDialogList() {
        val dlg = selectElderDlg ?: return
        val rv  = selectElderRecycler ?: return
        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)

        rv.adapter = ElderAdapter(
            elderList.map { it.contactId.toString() to it.display },

            onItemClick = { elder: Pair<String, String> ->
                val contactIdStr = elder.first
                val display = elder.second

                txtSelectedElder.text = display

                smartPref.edit()
                    .putString("elder_id", contactIdStr)
                    .putString("elder_name", display)
                    .apply()

                getSharedPreferences("app", MODE_PRIVATE).edit()
                    .putInt("user_id", userId)
                    .putString("contact_id", contactIdStr)
                    .apply()

                Log.d("SelectElder", "saved user_id=$userId, contact_id=$contactIdStr")
                Toast.makeText(this, "已選擇 $display", Toast.LENGTH_SHORT).show()
                dlg.dismiss()
            },

            onItemLongClick = l@{ elder: Pair<String, String> ->
                val cid = elder.first.toIntOrNull() ?: return@l
                val picked = elderList.firstOrNull { it.contactId == cid } ?: return@l

                val caregiverPhoneRaw = smartPref.getString("phone", "") ?: ""
                if (caregiverPhoneRaw.isBlank()) {
                    Toast.makeText(this, "無法取得登入者手機，無法刪除", Toast.LENGTH_SHORT).show()
                    return@l
                }

                val userPhoneNorm = normalizePhone(caregiverPhoneRaw)
                val contactPhoneNorm = normalizePhone(picked.phone)
                if (userPhoneNorm.isBlank() || contactPhoneNorm.isBlank()) {
                    Toast.makeText(this, "電話格式異常，無法刪除", Toast.LENGTH_SHORT).show()
                    return@l
                }

                AlertDialog.Builder(this)
                    .setTitle("刪除被照護者")
                    .setMessage("確定要刪除 ${picked.display} 嗎？")
                    .setPositiveButton("刪除") { _, _ ->
                        deleteContact(userPhoneNorm, contactPhoneNorm)
                    }
                    .setNegativeButton("取消", null)
                    .show()
            }
        )
}

    private fun normalizePhone(p: String?): String {
        if (p.isNullOrBlank()) return ""
        val digits = p.replace(Regex("[^0-9]"), "")
        return when {
            digits.startsWith("8869") && digits.length >= 12 -> "0" + digits.substring(3)
            digits.startsWith("886") && digits.length > 3    -> digits.substring(3)
            else -> digits
        }
    }

    private fun showAddElderDialog() {
        val inputView = layoutInflater.inflate(R.layout.dialog_add_elder, null)
        val editPhone = inputView.findViewById<EditText>(R.id.editContactPhone)

        AlertDialog.Builder(this)
            .setTitle("新增被照護者")
            .setView(inputView)
            .setPositiveButton("新增") { _, _ ->
                val raw = editPhone.text?.toString()?.trim().orEmpty()
                val phone = normalizePhone(raw)

                if (phone.length != 10 || !phone.startsWith("09")) {
                    Toast.makeText(this, "手機格式錯誤，需為09開頭共10碼", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }

                val self = normalizePhone(getSharedPreferences("smartcare_pref", MODE_PRIVATE).getString("phone", ""))
                if (phone == self) { Toast.makeText(this, "不可將自己設為被照護者", Toast.LENGTH_SHORT).show(); return@setPositiveButton }

                addNewElder(phone)
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

        RetrofitClient.apiService.getUser(phone)
            .enqueue(object : Callback<LoginResponse> {
                override fun onResponse(call: Call<LoginResponse>, response: Response<LoginResponse>) {
                    if (!response.isSuccessful) {
                        // 404 或錯誤訊息包含「找不到 / 不存在 / not found」→ 顯示「無此被照護者」
                        val raw = try { response.errorBody()?.string().orEmpty() } catch (_: Exception) { "" }
                        val isNotFound = response.code() == 404 ||
                                raw.contains("找不到") || raw.contains("不存在") ||
                                raw.contains("not found", ignoreCase = true)
                        val msg = if (isNotFound) "無此被照護者" else "查詢失敗：${response.code()}"
                        Toast.makeText(this@DashboardActivity, msg, Toast.LENGTH_SHORT).show()
                        return
                    }

                    val body = response.body()
                    if (body == null) {
                        Toast.makeText(this@DashboardActivity, "無此被照護者", Toast.LENGTH_SHORT).show()
                        return
                    }

                    val roleId = body.roleId ?: -1
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
                            override fun onResponse(call: Call<Map<String, String>>, resp: Response<Map<String, String>>) {
                                if (resp.isSuccessful) {
                                    Toast.makeText(this@DashboardActivity, "新增成功", Toast.LENGTH_SHORT).show()
                                    loadContacts()
                                } else {
                                    val raw = try { resp.errorBody()?.string().orEmpty() } catch (_: Exception) { "" }
                                    // 可選：若後端回 409 代表關係已存在，友善提示
                                    val msg = if (resp.code() == 409 ||
                                        raw.contains("已存在") || raw.contains("duplicate", ignoreCase = true)) {
                                        "已存在此被照護者關係"
                                    } else {
                                        try {
                                            val json = JSONObject(raw)
                                            json.optString("error", "新增失敗：${resp.code()}")
                                        } catch (_: Exception) {
                                            "新增失敗：${resp.code()}"
                                        }
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
                                Toast.makeText(this@DashboardActivity, "目前沒有設定被照護者", Toast.LENGTH_SHORT).show()
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

                            val existingCid = appPref.getString("contact_id", null)
                            if (elderList.isNotEmpty() && (existingCid.isNullOrBlank() || !existingCid.all(Char::isDigit))) {
                                val first = elderList.first()
                                appPref.edit().putString("contact_id", first.contactId.toString()).apply()

                                getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit()
                                    .putString("elder_id", first.contactId.toString())
                                    .putString("elder_name", first.display)
                                    .apply()

                                txtSelectedElder.text = first.display
                                Log.d("AutoPick", "Auto selected elder contact_id=${first.contactId} (${first.display})")
                                logPrefs("afterAutoPick")
                            }

                            val savedCidStr = smartPref.getString("elder_id", null)
                            val preferred = savedCidStr?.toIntOrNull()
                                ?.let { cid -> elderList.firstOrNull { it.contactId == cid } }

                            val chosen = preferred ?: elderList.first()

                            smartPref.edit()
                                .putString("elder_id", chosen.contactId.toString())
                                .putString("elder_name", chosen.display)
                                .apply()

                            appPref.edit()
                                .putInt("user_id", userId)
                                .putString("contact_id", chosen.contactId.toString())
                                .apply()

                            txtSelectedElder.text = chosen.display
                            logPrefs("loadContacts-autoSelect")
                        } else {
                            val raw = response.errorBody()?.string()
                            Log.e("DashboardDebug", "載入失敗 code=${response.code()}, body=$raw")
                            Toast.makeText(this@DashboardActivity, "載入失敗：${response.code()}", Toast.LENGTH_SHORT).show()
                        }
                    } catch (e: Exception) {
                        Log.e("DashboardDebug", "解析錯誤: ${e.message}", e)
                        Toast.makeText(this@DashboardActivity, "解析錯誤：${e.message}", Toast.LENGTH_LONG).show()
                    }
                }

                override fun onFailure(call: Call<List<EmergencyContact>>, t: Throwable) {
                    Log.e("DashboardDebug", "API 請求失敗: ${t.message}", t)
                    Toast.makeText(this@DashboardActivity, "連線錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    // ========================
    // 刪除（只用「正確順序」）
    // ========================
    private fun deleteContact(userPhone: String, contactPhone: String) {
        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val appPref = getSharedPreferences("app", MODE_PRIVATE)

        val currentCid = smartPref.getString("elder_id", null)
        val deletingItem = elderList.firstOrNull {
            it.phone.replace(Regex("[^0-9]"), "") == contactPhone.replace(Regex("[^0-9]"), "")
        }

        // 若刪掉的是目前選取的對象，先清掉偏好與 UI
        if (deletingItem != null && currentCid == deletingItem.contactId.toString()) {
            smartPref.edit().remove("elder_id").remove("elder_name").apply()
            appPref.edit().remove("contact_id").apply()
            txtSelectedElder.text = ""
            logPrefs("beforeDelete-clearSelected")
        }

        // 正常順序呼叫：user_phone=登入者、contact_phone=被刪除者
        val req = DeleteEmergencyContactRequest(
            user_phone = userPhone,
            contact_phone = contactPhone
        )
        Log.d("DeleteDebug", "Try DELETE /contact (normal) | user=${req.user_phone}, contact=${req.contact_phone}")

        RetrofitClient.apiService.deleteEmergencyContactContactBody(req)
            .enqueue(object : Callback<Map<String, String>> {
                override fun onResponse(
                    call: Call<Map<String, String>>,
                    resp: Response<Map<String, String>>
                ) {
                    if (resp.isSuccessful) { afterDeleteSuccess(contactPhone); return }

                    val raw = try { resp.errorBody()?.string().orEmpty() } catch (_: Exception) { "" }
                    Log.d("DeleteDebug", "DELETE resp code=${resp.code()} raw=$raw")

                    // 後端雙向刪除：第一向已刪成功、第二向不存在 → 500，訊息含關鍵字就視為成功
                    if (resp.code() == 500 && (raw.contains("照護關係可刪除") ||
                                (raw.contains("找不到") && raw.contains("照護關係")))) {
                        Log.d("DeleteDebug", "Treat 500(NotFound second direction) as success")
                        afterDeleteSuccess(contactPhone); return
                    }

                    // 🔁 ngrok/代理問題 → 改打 POST 後備
                    if (resp.code() == 503 || raw.contains("ngrok", ignoreCase = true)) {
                        fallbackPostDelete(req, contactPhone); return
                    }

                    Toast.makeText(this@DashboardActivity, "刪除失敗：${resp.code()}", Toast.LENGTH_SHORT).show()
                }

                override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                    Log.w("DeleteDebug", "DELETE onFailure: ${t.message}")
                    // 網路層失敗也嘗試一次後備
                    fallbackPostDelete(req, contactPhone)
                }
            })
    }

    private fun fallbackPostDelete(req: DeleteEmergencyContactRequest, contactPhone: String) {
        Log.d("DeleteDebug", "Try POST /contact/delete (fallback) | user=${req.user_phone}, contact=${req.contact_phone}")
        RetrofitClient.apiService.deleteEmergencyContactContactPost(req)
            .enqueue(object : Callback<Map<String, String>> {
                override fun onResponse(call: Call<Map<String, String>>, resp: Response<Map<String, String>>) {
                    if (resp.isSuccessful) {
                        afterDeleteSuccess(contactPhone)
                    } else {
                        Toast.makeText(this@DashboardActivity, "刪除失敗：${resp.code()}", Toast.LENGTH_SHORT).show()
                    }
                }
                override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                    Toast.makeText(this@DashboardActivity, "連線錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }

    private fun afterDeleteSuccess(contactPhone: String) {
        val removed = elderList.removeAll {
            it.phone.replace("[^0-9]".toRegex(), "") ==
                    contactPhone.replace("[^0-9]".toRegex(), "")
        }
        Log.d("DeleteDebug", "本地已移除: $removed")
        Toast.makeText(this@DashboardActivity, "刪除成功", Toast.LENGTH_SHORT).show()

        if (selectElderDlg?.isShowing == true) {
            rebuildElderDialogList()
        }
        loadContacts()
    }
}
