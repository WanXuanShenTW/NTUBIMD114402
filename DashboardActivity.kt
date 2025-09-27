package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.media3.common.util.UnstableApi
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.ElderAdapter
import com.example.myapplication.model.ApiResponse
import com.example.myapplication.model.UserDto
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

@UnstableApi
class DashboardActivity : AppCompatActivity() {

    data class ElderItem(
        val elderUserId: Int,
        val phone: String,
        val display: String
    )

    private var userId: Int = -1
    private lateinit var txtSelectedElder: TextView
    private var elderList = mutableListOf<ElderItem>()
    private var selectElderDlg: AlertDialog? = null
    private var selectElderRecycler: RecyclerView? = null
    @Volatile private var loadingContacts = false

    override fun onCreate(savedInstanceState: Bundle?) {
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

        restoreElderSelection()

        txtSelectedElder.setOnClickListener {
            if (elderList.isEmpty()) {
                Toast.makeText(this, "尚未載入被照護者", Toast.LENGTH_SHORT).show()
            } else {
                showSelectElderDialog()
            }
        }

        loadContacts("onCreate")

        findViewById<ImageButton>(R.id.navEmergency).setOnClickListener {
            val selected = getSelectedElder()  // 下面第2點會新增這個方法
            val phone = selected?.phone ?: ""

            if (phone.isBlank()) {
                Toast.makeText(this, "請先選擇被照護者", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // 順手存一份到 app 偏好，讓目標頁面可備援讀取
            getSharedPreferences("app", MODE_PRIVATE)
                .edit()
                .putString("elder_phone", phone)
                .apply()

            startActivity(
                Intent(this, EmergencyContactsActivity::class.java)
                    .putExtra("elder_phone", phone)
            )
        }

        findViewById<ImageButton>(R.id.btnLivingRoom).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }
        findViewById<ImageButton>(R.id.navChat).setOnClickListener {
            startActivity(Intent(this, VoiceResultActivity::class.java))
        }
        findViewById<ImageButton>(R.id.navSearch).setOnClickListener {
            startActivity(Intent(this, VideoListActivity::class.java))
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
            "$where | app.user_id=${a.getInt("user_id", -1)}, app.elder_id=${a.getInt("elder_id", -1)} ; " +
                    "smart.phone=${s.getString("phone", null)}, smart.elder_id=${s.getString("elder_id", null)}, " +
                    "smart.elder_name=${s.getString("elder_name", null)}"
        )
    }

    private fun getSelectedElder(): ElderItem? {
        val appPref = getSharedPreferences("app", MODE_PRIVATE)
        val eid = appPref.getInt("elder_id", -1)
        return elderList.firstOrNull { it.elderUserId == eid } ?: elderList.firstOrNull()
    }

    private fun showSelectElderDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_select_elder, null)
        val dialogRecycler = dialogView.findViewById<RecyclerView>(R.id.dialogRecyclerElder)

        dialogRecycler.layoutManager = LinearLayoutManager(this)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        // 存參考以便重建列表
        selectElderDlg = dialog
        selectElderRecycler = dialogRecycler
        rebuildElderDialogList()

        dialog.show()
    }

    private fun rebuildElderDialogList() {
        val dlg = selectElderDlg ?: return
        val rv  = selectElderRecycler ?: return

        rv.adapter = ElderAdapter(
            elderList.map { it.elderUserId.toString() to it.display },

            onItemClick = { elder: Pair<String, String> ->
                val clicked = elderList.firstOrNull { it.elderUserId.toString() == elder.first }
                    ?: return@ElderAdapter
                onPickElder(clicked, emitBroadcast = true, showToast = true)
                dlg.dismiss()
            },

            onItemLongClick = { /* no-op */ }
        )
    }

    // 取代原本的 onPickElder
    private fun onPickElder(
        item: ElderItem,
        emitBroadcast: Boolean = true,
        showToast: Boolean = true
    ) {
        // app 偏好
        getSharedPreferences("app", MODE_PRIVATE).edit()
            .putInt("elder_id", item.elderUserId)
            .putString("elder_name", item.display)
            .putString("elder_phone", item.phone)
            .apply()

        // 兼容舊 smartcare_pref（可保留）
        getSharedPreferences("smartcare_pref", MODE_PRIVATE).edit()
            .putString("elder_id", item.elderUserId.toString())
            .putString("elder_name", item.display)
            .apply()

        // UI
        txtSelectedElder.text = item.display
        logPrefs("afterPick")

        if (emitBroadcast) {
            sendBroadcast(Intent("com.example.myapplication.ACTION_ELDER_CHANGED").apply {
                setPackage(packageName)
                putExtra("elder_id", item.elderUserId)
                putExtra("elder_name", item.display)
            })
        }
        if (showToast) {
            Toast.makeText(this, "已選擇 ${item.display}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun restoreElderSelection() {
        val app = getSharedPreferences("app", MODE_PRIVATE)
        val name = app.getString("elder_name", null)
        if (!name.isNullOrBlank()) {
            txtSelectedElder.text = name
            return
        }
        val smart = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val name2 = smart.getString("elder_name", null)
        txtSelectedElder.text = name2?.takeIf { it.isNotBlank() } ?: "尚未選擇被照護者"
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

    private fun loadContacts(origin: String = "unknown") {
        if (loadingContacts) {
            Log.d("ContactsDebug", "skip duplicate loadContacts (origin=$origin)")
            return
        }
        loadingContacts = true

        val smartPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val appPref   = getSharedPreferences("app", MODE_PRIVATE)

        val rawPhone = smartPref.getString("phone", "") ?: ""
        val caregiverPhone = normalizePhone(rawPhone)

        if (caregiverPhone.isBlank()) {
            Toast.makeText(this, "無法取得登入者手機，請重新登入", Toast.LENGTH_SHORT).show()
            Log.e("ContactsDebug", "smartcare_pref.phone 為空，無法呼叫 /contacts/caregiver (raw='$rawPhone'), origin=$origin")
            loadingContacts = false
            return
        }

        Log.d("ContactsDebug", "→ GET /contacts/caregiver?caregiver_phone=$caregiverPhone (raw='$rawPhone'), origin=$origin")

        RetrofitClient.apiService.getCareReceiversByCaregiverPhone(caregiverPhone)
            .enqueue(object : Callback<ApiResponse<List<UserDto>>> {
                override fun onResponse(
                    call: Call<ApiResponse<List<UserDto>>>,
                    resp: Response<ApiResponse<List<UserDto>>>
                ) {
                    try {
                        val code = resp.code()
                        val body = resp.body()
                        val data = body?.data

                        Log.d("ContactsDebug", "HTTP=$code, api.code=${body?.code}, msg=${body?.message}, data.size=${data?.size ?: -1}, origin=$origin")

                        if (!resp.isSuccessful) {
                            val rawErr = resp.errorBody()?.string()
                            Log.e("ContactsDebug", "errorBody=$rawErr, origin=$origin")
                            Toast.makeText(this@DashboardActivity, "載入失敗：$code", Toast.LENGTH_SHORT).show()
                            return
                        }

                        if ((body?.code ?: 200) != 200) {
                            Toast.makeText(this@DashboardActivity, body?.message ?: "查詢失敗", Toast.LENGTH_SHORT).show()
                            return
                        }

                        elderList.clear()
                        val users = data ?: emptyList()

                        users.forEach {
                            Log.d("ContactsDebug", "user_id=${it.userId}, name='${it.name}', phone='${it.phone}', role_id=${it.roleId}, origin=$origin")
                        }

                        if (users.isEmpty()) {
                            Toast.makeText(this@DashboardActivity, "目前沒有設定被照護者", Toast.LENGTH_SHORT).show()
                            appPref.edit().remove("elder_id").remove("elder_name").apply()
                            smartPref.edit().remove("elder_id").remove("elder_name").apply()
                            txtSelectedElder.text = "尚未選擇被照護者"
                            logPrefs("loadContacts-empty")
                            return
                        }

                        elderList.addAll(
                            users.map { u ->
                                val display = (if (u.name.isNotBlank()) u.name else "尾碼 ${u.phone.takeLast(4)}") +
                                        " - ${u.phone}"
                                ElderItem(elderUserId = u.userId, phone = u.phone, display = display)
                            }
                        )

                        val savedSmartElderId = smartPref.getString("elder_id", null)?.toIntOrNull()
                        val selected: ElderItem =
                            if (savedSmartElderId != null && savedSmartElderId > 0) {
                                elderList.firstOrNull { it.elderUserId == savedSmartElderId } ?: elderList.first()
                            } else {
                                elderList.first()
                            }

                        val oldAppEid = appPref.getInt("elder_id", -1)
                        val changed = oldAppEid != selected.elderUserId

                        onPickElder(
                            selected,
                            emitBroadcast = changed,
                            showToast = false
                        )
                    } finally {
                        loadingContacts = false
                    }
                }

                override fun onFailure(call: Call<ApiResponse<List<UserDto>>>, t: Throwable) {
                    Log.e("ContactsDebug", "onFailure: ${t.message}, origin=$origin", t)
                    Toast.makeText(this@DashboardActivity, "連線錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                    loadingContacts = false
                }
            })
    }
}
