package com.example.myapplication

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.adapter.EmergencyContactAdapter
import com.example.myapplication.model.ApiResponse
import com.example.myapplication.model.CaregiverContactDto
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import android.view.Menu
import android.view.MenuItem
import android.widget.EditText
import androidx.appcompat.app.AlertDialog
import com.example.myapplication.model.CreateContactReq
import com.example.myapplication.model.DeleteContactReq
import androidx.recyclerview.widget.ItemTouchHelper
import com.example.myapplication.model.DeleteContactByIdReq


class EmergencyContactsActivity : AppCompatActivity() {

    private lateinit var recycler: RecyclerView
    private lateinit var progress: ProgressBar
    private lateinit var emptyView: TextView
    private lateinit var subtitle: TextView
    private lateinit var adapter: EmergencyContactAdapter
    private var elderPhone: String = ""
    private var elderId: Int? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emergency_contacts)

        recycler = findViewById(R.id.recycler)
        progress = findViewById(R.id.progress)
        emptyView = findViewById(R.id.emptyView)
        subtitle = findViewById(R.id.subtitle)

        findViewById<ImageView>(R.id.backButton).setOnClickListener { finish() }

        findViewById<ImageView>(R.id.btnAddContact).setOnClickListener {
            showAddContactDialog()
        }

        recycler.layoutManager = LinearLayoutManager(this)

        // 點電話圖示 → 開啟撥號器
        adapter = EmergencyContactAdapter { phone ->
            val p = phone?.trim().orEmpty()
            if (p.isEmpty()) {
                Toast.makeText(this, "此聯絡人沒有電話", Toast.LENGTH_SHORT).show()
            } else {
                startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:${Uri.encode(p)}")))
            }
        }
        recycler.adapter = adapter
        attachSwipeToDelete()

        val sp = getSharedPreferences("app", MODE_PRIVATE)

        elderId = intent.getIntExtra("elder_id", -1).takeIf { it > 0 }
            ?: sp.getInt("elder_id", -1).takeIf { it > 0 }

        elderPhone = intent.getStringExtra("elder_phone")
            ?.takeIf { it.isNotBlank() }
            ?: sp.getString("elder_phone", null)?.takeIf { !it.isNullOrBlank() }
                    ?: ""

        if (elderPhone.isBlank()) {
            Toast.makeText(this, "沒有找到長者電話，請先在主畫面選擇長者", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        subtitle.text = buildString {
            append("長者電話：$elderPhone")
        }

        fetchContacts(elderPhone)
    }


    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_emergency_contacts, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_add_contact -> {
                showAddContactDialog()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showAddContactDialog() {
        val view = layoutInflater.inflate(R.layout.dialog_add_contact, null)
        val edtPhone = view.findViewById<EditText>(R.id.edtCaregiverPhone)
        val edtRel   = view.findViewById<EditText>(R.id.edtRelationship)

        AlertDialog.Builder(this)
            .setTitle("新增緊急連絡人")
            .setView(view)
            .setNegativeButton("取消", null)
            .setPositiveButton("新增") { _, _ ->
                val phone = edtPhone.text.toString().trim()
                val relationship = edtRel.text.toString().trim().ifBlank { "未填寫" } // 你要的預設可改

                if (elderPhone.isBlank()) {
                    toast("找不到長者電話，請先在首頁選擇長者")
                    return@setPositiveButton
                }
                if (phone.isBlank()) {
                    toast("請輸入照護者電話")
                    return@setPositiveButton
                }
                createContact(phone, relationship)
            }
            .show()
    }

    private fun createContact(caregiverPhone: String, relationship: String) {
        showLoading(true)
        val body = CreateContactReq(
            elder_phone = elderPhone,
            caregiver_phone = caregiverPhone,
            relationship = relationship
        )
        RetrofitClient.apiService.createContact(body)
            .enqueue(object : Callback<ApiResponse<Unit>> {
                override fun onResponse(
                    call: Call<ApiResponse<Unit>>,
                    response: Response<ApiResponse<Unit>>
                ) {
                    showLoading(false)
                    val b = response.body()
                    if (response.isSuccessful && b != null && b.code == 200) {
                        Toast.makeText(this@EmergencyContactsActivity, "新增成功", Toast.LENGTH_SHORT).show()
                        fetchContacts(elderPhone) // 重新載入清單
                    } else {
                        val errStr = response.errorBody()?.string().orEmpty()
                        Log.e("EC", "POST /contact FAILED http=${response.code()}, apiMsg=${b?.message}, body=$errStr")
                        Toast.makeText(
                            this@EmergencyContactsActivity,
                            b?.message ?: "新增失敗（HTTP ${response.code()}）",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onFailure(call: Call<ApiResponse<Unit>>, t: Throwable) {
                    showLoading(false)
                    Log.e("EC", "POST /contact NETWORK FAIL: ${t.message}\n${t.stackTraceToString()}")
                    Toast.makeText(this@EmergencyContactsActivity, "網路錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                }
            })
}


    private fun toast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    private fun fetchContacts(elderPhone: String) {
        showLoading(true)
        RetrofitClient.apiService.getCaregiversByElder(elderPhone)
            .enqueue(object : Callback<ApiResponse<List<CaregiverContactDto>>> {
                override fun onResponse(
                    call: Call<ApiResponse<List<CaregiverContactDto>>>,
                    response: Response<ApiResponse<List<CaregiverContactDto>>>
                ) {
                    showLoading(false)
                    val body = response.body()
                    if (response.isSuccessful && body != null && body.code == 200) {
                        val list = body.data ?: emptyList()
                        if (list.isEmpty()) {
                            showEmpty(true)
                            adapter.submitList(emptyList()) // 清舊資料
                        } else {
                            showEmpty(false)
                            adapter.submitList(list)
                        }
                    } else {
                        showEmpty(true)
                        adapter.submitList(emptyList())
                        Toast.makeText(
                            this@EmergencyContactsActivity,
                            body?.message ?: "查詢失敗",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onFailure(
                    call: Call<ApiResponse<List<CaregiverContactDto>>>,
                    t: Throwable
                ) {
                    showLoading(false)
                    showEmpty(true)
                    adapter.submitList(emptyList())
                    Toast.makeText(
                        this@EmergencyContactsActivity,
                        "網路錯誤：${t.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            })
    }

    private fun attachSwipeToDelete() {
        val swipeCb = object : ItemTouchHelper.SimpleCallback(0, ItemTouchHelper.LEFT) {
            override fun onMove(
                rv: RecyclerView, vh: RecyclerView.ViewHolder, target: RecyclerView.ViewHolder
            ): Boolean = false

            override fun onSwiped(vh: RecyclerView.ViewHolder, dir: Int) {
                val pos = vh.bindingAdapterPosition
                val item = adapter.getItem(pos)
                // 左滑後先彈出確認，若取消就復原該列
                confirmDelete(item, pos)
            }
        }
        ItemTouchHelper(swipeCb).attachToRecyclerView(recycler)
    }

    private fun confirmDelete(item: CaregiverContactDto, pos: Int) {
        AlertDialog.Builder(this)
            .setTitle("刪除緊急連絡人")
            .setMessage("確定要刪除「${item.name ?: item.caregiverPhone}」嗎？")
            .setNegativeButton("取消") { _, _ ->
                // 使用者取消：把被左滑的那列復原
                adapter.notifyItemChanged(pos)
            }
            .setPositiveButton("刪除") { _, _ ->
                deleteContact(item, pos)
            }
            .setOnCancelListener {
                adapter.notifyItemChanged(pos)
            }
            .show()
    }

    private fun deleteContact(item: CaregiverContactDto, pos: Int) {
        if (pos == RecyclerView.NO_POSITION) return
        showLoading(true)

        val cgPhone = item.caregiverPhone
        if (elderPhone.isBlank() || cgPhone.isNullOrBlank()) {
            Toast.makeText(this, "缺少電話資訊，無法刪除", Toast.LENGTH_SHORT).show()
            adapter.notifyItemChanged(pos)
            showLoading(false)
            return
        }

        val req = DeleteContactReq(
            elder_phone = elderPhone,
            caregiver_phone = cgPhone
        )
        Log.d("EC", "DELETE /contact body = $req")

        RetrofitClient.apiService.deleteContact(req)
            .enqueue(object : retrofit2.Callback<ApiResponse<Unit>> {
                override fun onResponse(
                    call: retrofit2.Call<ApiResponse<Unit>>,
                    response: retrofit2.Response<ApiResponse<Unit>>
                ) {
                    showLoading(false)
                    val b = response.body()
                    if (response.isSuccessful && b != null && b.code == 200) {
                        Toast.makeText(this@EmergencyContactsActivity, "已刪除", Toast.LENGTH_SHORT).show()
                        adapter.removeAt(pos)
                        emptyView.visibility = if (adapter.itemCount == 0) View.VISIBLE else View.GONE
                    } else {
                        val errStr = response.errorBody()?.string().orEmpty()
                        Log.e("EC", "DELETE /contact FAILED http=${response.code()}, apiMsg=${b?.message}, body=$errStr")
                        Toast.makeText(
                            this@EmergencyContactsActivity,
                            "刪除失敗（HTTP ${response.code()}）${if (errStr.isNotBlank()) "：$errStr" else ""}",
                            Toast.LENGTH_SHORT
                        ).show()
                        adapter.notifyItemChanged(pos)
                    }
                }

                override fun onFailure(
                    call: retrofit2.Call<ApiResponse<Unit>>,
                    t: Throwable
                ) {
                    showLoading(false)
                    Log.e("EC", "DELETE /contact NETWORK FAIL: ${t.message}\n${t.stackTraceToString()}")
                    Toast.makeText(this@EmergencyContactsActivity, "網路錯誤：${t.message}", Toast.LENGTH_SHORT).show()
                    adapter.notifyItemChanged(pos)
                }
            })
    }

    private fun deleteCallback(pos: Int) = object : retrofit2.Callback<ApiResponse<Unit>> {
        override fun onResponse(
            call: retrofit2.Call<ApiResponse<Unit>>,
            response: retrofit2.Response<ApiResponse<Unit>>
        ) {
            showLoading(false)
            val b = response.body()
            if (response.isSuccessful && b != null && b.code == 200) {
                Toast.makeText(this@EmergencyContactsActivity, "已刪除", Toast.LENGTH_SHORT).show()
                adapter.removeAt(pos)
                emptyView.visibility = if (adapter.itemCount == 0) View.VISIBLE else View.GONE
            } else {
                val errStr = response.errorBody()?.string().orEmpty()
                Log.e("EC", "DELETE FAILED http=${response.code()}, apiMsg=${b?.message}, body=$errStr")
                Toast.makeText(
                    this@EmergencyContactsActivity,
                    "刪除失敗（HTTP ${response.code()}）${if (errStr.isNotBlank()) "：$errStr" else ""}",
                    Toast.LENGTH_SHORT
                ).show()
                adapter.notifyItemChanged(pos)
            }
        }

        override fun onFailure(call: retrofit2.Call<ApiResponse<Unit>>, t: Throwable) {
            showLoading(false)
            Log.e("EC", "DELETE NETWORK FAIL: ${t.message}\n${t.stackTraceToString()}")
            Toast.makeText(this@EmergencyContactsActivity, "網路錯誤：${t.message}", Toast.LENGTH_SHORT).show()
            adapter.notifyItemChanged(pos)
        }
    }

    private fun showEmpty(isEmpty: Boolean) {
        emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        recycler.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun showLoading(loading: Boolean) {
        progress.visibility = if (loading) View.VISIBLE else View.GONE
    }
}
