package com.example.myapplication

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatButton
import androidx.media3.common.util.UnstableApi
import com.example.myapplication.model.*
import com.example.myapplication.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import com.example.myapplication.AppKeys

@UnstableApi
class EditProfileActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var spinnerRole: Spinner
    private lateinit var textPhone: TextView
    private lateinit var spinnerCity: Spinner
    private lateinit var spinnerDistrict: Spinner
    private lateinit var editAddressLine: EditText
    private lateinit var btnSave: Button
    private lateinit var btnCancel: ImageView
    private lateinit var btnEdit: TextView
    private lateinit var btnChangePassword: Button
    private lateinit var btnDeleteAccount: Button
    private lateinit var btnAddLine: AppCompatButton
    private lateinit var loadingProgress: ProgressBar
    private var suppressCityListener = false

    private var currentPhone: String = ""

    // 編輯狀態 & 原始值（用於取消還原）
    private var isEditing = false
    private var originalName = ""
    private var originalRolePos = 0
    private var originalAddress = ""

    private val districtsMap = linkedMapOf(
        "臺北市" to listOf("中正區","大同區","中山區","松山區","大安區","萬華區","信義區","士林區","北投區","內湖區","南港區","文山區"),
        "新北市" to listOf("板橋區","新莊區","中和區","永和區","土城區","樹林區","三峽區","鶯歌區","三重區","蘆洲區","五股區","泰山區","林口區","八里區","淡水區","三芝區","石門區","金山區","萬里區","汐止區","瑞芳區","貢寮區","雙溪區","平溪區","新店區","深坑區","石碇區","坪林區","烏來區"),
        "桃園市" to listOf("桃園區","中壢區","平鎮區","八德區","楊梅區","蘆竹區","大溪區","龍潭區","龜山區","大園區","觀音區","新屋區","復興區"),
        "臺中市" to listOf("中區","東區","南區","西區","北區","北屯區","西屯區","南屯區","太平區","大里區","霧峰區","烏日區","豐原區","后里區","東勢區","石岡區","新社區","和平區","神岡區","潭子區","大雅區","大肚區","龍井區","沙鹿區","梧棲區","清水區","大甲區","外埔區","大安區"),
        "臺南市" to listOf("中西區","東區","南區","北區","安平區","安南區","永康區","歸仁區","新化區","左鎮區","玉井區","楠西區","南化區","仁德區","關廟區","龍崎區","官田區","麻豆區","佳里區","西港區","七股區","將軍區","學甲區","北門區","新營區","後壁區","白河區","東山區","六甲區","下營區","柳營區","鹽水區","善化區","大內區","山上區","新市區","安定區"),
        "高雄市" to listOf("楠梓區","左營區","鼓山區","三民區","鹽埕區","前金區","新興區","苓雅區","前鎮區","旗津區","小港區","鳳山區","林園區","大寮區","大樹區","大社區","仁武區","鳥松區","岡山區","橋頭區","燕巢區","田寮區","阿蓮區","路竹區","湖內區","茄萣區","永安區","彌陀區","梓官區","旗山區","美濃區","六龜區","甲仙區","杉林區","內門區","茂林區","桃源區","那瑪夏區"),
        "基隆市" to listOf("仁愛區","信義區","中正區","中山區","安樂區","暖暖區","七堵區"),
        "新竹市" to listOf("東區","北區","香山區"),
        "嘉義市" to listOf("東區","西區"),
        "新竹縣" to listOf("竹北市","竹東鎮","新埔鎮","關西鎮","湖口鄉","新豐鄉","芎林鄉","橫山鄉","北埔鄉","寶山鄉","峨眉鄉","尖石鄉","五峰鄉"),
        "苗栗縣" to listOf("苗栗市","頭份市","竹南鎮","後龍鎮","通霄鎮","苑裡鎮","卓蘭鎮","西湖鄉","頭屋鄉","公館鄉","銅鑼鄉","三義鄉","造橋鄉","三灣鄉","南庄鄉","大湖鄉","獅潭鄉","泰安鄉"),
        "彰化縣" to listOf("彰化市","員林市","鹿港鎮","和美鎮","北斗鎮","溪湖鎮","田中鎮","二林鎮","線西鄉","伸港鄉","福興鄉","秀水鄉","花壇鄉","芬園鄉","大村鄉","永靖鄉","社頭鄉","埔心鄉","埔鹽鄉","溪州鄉","田尾鄉","埤頭鄉","竹塘鄉","大城鄉","芳苑鄉","二水鄉"),
        "南投縣" to listOf("南投市","埔里鎮","草屯鎮","竹山鎮","集集鎮","名間鄉","鹿谷鄉","中寮鄉","魚池鄉","國姓鄉","水里鄉","信義鄉","仁愛鄉"),
        "雲林縣" to listOf("斗六市","斗南鎮","虎尾鎮","西螺鎮","土庫鎮","北港鎮","莿桐鄉","林內鄉","二崙鄉","崙背鄉","麥寮鄉","東勢鄉","褒忠鄉","臺西鄉","元長鄉","四湖鄉","口湖鄉","水林鄉","古坑鄉","大埤鄉"),
        "嘉義縣" to listOf("太保市","朴子市","布袋鎮","大林鎮","民雄鄉","溪口鄉","新港鄉","六腳鄉","東石鄉","義竹鄉","鹿草鄉","水上鄉","中埔鄉","竹崎鄉","梅山鄉","番路鄉","大埔鄉","阿里山鄉"),
        "屏東縣" to listOf("屏東市","潮州鎮","東港鎮","恆春鎮","萬丹鄉","長治鄉","麟洛鄉","九如鄉","里港鄉","鹽埔鄉","高樹鄉","萬巒鄉","內埔鄉","竹田鄉","新埤鄉","枋寮鄉","新園鄉","崁頂鄉","林邊鄉","南州鄉","佳冬鄉","琉球鄉","車城鄉","滿州鄉","枋山鄉","三地門鄉","霧臺鄉","瑪家鄉","泰武鄉","來義鄉","春日鄉","獅子鄉","牡丹鄉"),
        "宜蘭縣" to listOf("宜蘭市","羅東鎮","蘇澳鎮","頭城鎮","礁溪鄉","壯圍鄉","員山鄉","五結鄉","冬山鄉","三星鄉","大同鄉","南澳鄉"),
        "花蓮縣" to listOf("花蓮市","鳳林鎮","玉里鎮","新城鄉","吉安鄉","壽豐鄉","秀林鄉","光復鄉","豐濱鄉","瑞穗鄉","富里鄉","萬榮鄉","卓溪鄉"),
        "臺東縣" to listOf("臺東市","成功鎮","關山鎮","卑南鄉","鹿野鄉","池上鄉","東河鄉","長濱鄉","太麻里鄉","大武鄉","綠島鄉","蘭嶼鄉","延平鄉","海端鄉","金峰鄉","達仁鄉"),
        "澎湖縣" to listOf("馬公市","湖西鄉","白沙鄉","西嶼鄉","望安鄉","七美鄉"),
        "金門縣" to listOf("金城鎮","金湖鎮","金沙鎮","金寧鄉","烈嶼鄉","烏坵鄉"),
        "連江縣" to listOf("南竿鄉","北竿鄉","莒光鄉","東引鄉")
    )

    private val PLACEHOLDER = "請選擇"

    private val cities: List<String> = listOf(PLACEHOLDER) + districtsMap.keys.toList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_edit_profile)

        Log.d("Lifecycle", "EditProfileActivity onCreate 呼叫中")

        editName = findViewById(R.id.editName)
        spinnerRole = findViewById(R.id.spinnerRole)
        textPhone = findViewById(R.id.textPhone)
        spinnerCity = findViewById(R.id.spinnerCity)
        spinnerDistrict = findViewById(R.id.spinnerDistrict)
        editAddressLine = findViewById(R.id.editAddressLine)
        btnSave = findViewById(R.id.btnSave)
        btnCancel = findViewById(R.id.btnCancel)
        btnEdit = findViewById(R.id.btnEdit)
        btnChangePassword = findViewById(R.id.btnChangePassword)
        btnDeleteAccount = findViewById(R.id.btnDeleteAccount)
        btnAddLine = findViewById(R.id.btnAddLine)
        loadingProgress = findViewById(R.id.loadingProgress)

        val sharedPref = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        currentPhone = sharedPref.getString("phone", "") ?: ""
        Log.d("ProfileDebug", "sharedPref phone=$currentPhone")

        val currentName = sharedPref.getString("username", "")
        val currentAddress = sharedPref.getString("address", "")
        editName.setText(currentName)
        editAddressLine.setText(currentAddress)
        textPhone.text = currentPhone

        // 角色下拉
        val roleAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, listOf("家屬", "醫護人員"))
        roleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerRole.adapter = roleAdapter
        spinnerRole.isEnabled = false

        // 地址下拉
        setupAddressSpinners()

        // 初始狀態：非編輯
        toggleEditMode(false)
        snapshotCurrentFieldsAsOriginal()

        // 以伺服器為準載入使用者資料
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

                        val rolePosition = when (u.roleId) {
                            2 -> 0 // 家屬
                            3 -> 1 // 醫護人員
                            else -> 0
                        }
                        spinnerRole.setSelection(rolePosition)

                        // 回填地址
                        fillBackAddress(u.address)

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

        // ===== 事件 =====

        // 右上角「+ 編輯」
        btnEdit.setOnClickListener {
            isEditing = true
            toggleEditMode(true)
        }

        // 左上角返回：編輯中→還原；否則關閉
        btnCancel.setOnClickListener {
            if (isEditing) {
                restoreOriginalFields()
                isEditing = false
                toggleEditMode(false)
            } else {
                finish()
            }
        }

        // 儲存
        btnSave.setOnClickListener {
            val newName = editName.text.toString().trim()
            val newRoleId = when (spinnerRole.selectedItemPosition) { 0 -> 2; 1 -> 3; else -> 2 }
            val city = spinnerCity.selectedItem?.toString().orEmpty()
            val dist = spinnerDistrict.selectedItem?.toString().orEmpty()
            val line = editAddressLine.text.toString().trim()
            val fullAddress = listOf(city, dist, line)
                .filter { it.isNotBlank() && it != PLACEHOLDER }
                .joinToString("")

            if (newName.isEmpty()) {
                Toast.makeText(this, "姓名不可為空", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            loadingProgress.visibility = View.VISIBLE
            btnSave.isEnabled = false

            val req = UpdateUserRequest(
                phone   = currentPhone,
                name    = newName,
                roleId  = newRoleId,
                address = fullAddress.ifBlank { null }
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

                            val cityToSave = if (city == PLACEHOLDER) "" else city
                            val distToSave = if (dist == PLACEHOLDER) "" else dist

                            // 同步本地快取
                            getSharedPreferences("smartcare_pref", MODE_PRIVATE)
                                .edit()
                                .putString("username", newName)
                                .putString("address", fullAddress)
                                .putString("addr_city", cityToSave)
                                .putString("addr_dist", distToSave)
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

        btnAddLine.setOnClickListener {
            openLineOfficialAccount(AppKeys.LINE_OA_ID)
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

                    val rolePosition = when (u.roleId) {
                        2 -> 0 // 家屬
                        3 -> 1 // 醫護人員
                        else -> 0
                    }
                    spinnerRole.setSelection(rolePosition)

                    fillBackAddress(u.address)

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
        spinnerCity.isEnabled = isEditable
        spinnerDistrict.isEnabled = isEditable
        editAddressLine.isEnabled = isEditable

        btnSave.visibility = if (isEditable) View.VISIBLE else View.GONE
        btnSave.isEnabled = isEditable
        btnEdit.visibility = if (isEditable) View.GONE else View.VISIBLE

        // 這兩個功能建議一直可用
        btnChangePassword.visibility = View.VISIBLE
        btnChangePassword.isEnabled = true
        btnDeleteAccount.visibility = View.VISIBLE
        btnDeleteAccount.isEnabled = true
    }

    // 記錄目前欄位作為「原始值」
    private fun snapshotCurrentFieldsAsOriginal() {
        originalName = editName.text?.toString() ?: ""
        originalRolePos = spinnerRole.selectedItemPosition
        originalAddress = editAddressLine.text?.toString() ?: ""
    }

    // 還原到「原始值」
    private fun restoreOriginalFields() {
        editName.setText(originalName)
        spinnerRole.setSelection(originalRolePos)
        editAddressLine.setText(originalAddress)
    }

    private fun fillBackAddress(address: String?) {
        val addr = address.orEmpty()

        // 1) 先用本地快取（有的話最準）
        val sp = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
        val savedCityRaw = sp.getString("addr_city", "") ?: ""
        val savedDistRaw = sp.getString("addr_dist", "") ?: ""

        if (savedCityRaw.isNotBlank()) {
            // 選城市（容忍 台/臺）＋ 主動綁地區清單（不要等 onItemSelected）
            suppressCityListener = true
            selectSpinnerByText(spinnerCity, savedCityRaw)
            bindDistrictsForCity(savedCityRaw)
            suppressCityListener = false

            // 若快取沒有地區，就從整串地址猜地區
            val distToUse = if (savedDistRaw.isNotBlank()) {
                savedDistRaw
            } else {
                guessDistrictForCity(savedCityRaw, addr) ?: ""
            }

            spinnerDistrict.post {
                if (distToUse.isNotBlank()) {
                    selectSpinnerByText(spinnerDistrict, distToUse)

                    // 若原本沒快取地區，補存一份
                    if (savedDistRaw.isBlank()) {
                        sp.edit()
                            .putString("addr_dist", distToUse)
                            .apply()
                    }
                } else {
                    // 猜不到就維持「請選擇」
                    spinnerDistrict.setSelection(0)
                }

                // 剝掉「城市 +（可能的）地區」前綴，避免重複
                val normAddr = addr.replace("臺", "台")
                val prefix   = (savedCityRaw + distToUse).replace("臺", "台")
                val detail = if (prefix.isNotBlank() && normAddr.startsWith(prefix)) {
                    normAddr.removePrefix(prefix).trim()
                } else if (normAddr.startsWith(savedCityRaw.replace("臺", "台"))) {
                    // 至少把城市剝掉
                    normAddr.removePrefix(savedCityRaw.replace("臺", "台")).trim()
                } else {
                    addr
                }
                editAddressLine.setText(detail)
            }
            return
        }

        // 2) 沒快取 → 用字典試著切（只為了第一次顯示好看，非強制）
        val parsed = parseAddressByDicts(addr)
        if (parsed != null) {
            val (city, dist, rest) = parsed

            // 快取起來
            sp.edit()
                .putString("addr_city", city)
                .putString("addr_dist", dist)
                .apply()

            suppressCityListener = true
            selectSpinnerByText(spinnerCity, city)
            bindDistrictsForCity(city)
            suppressCityListener = false

            spinnerDistrict.post {
                selectSpinnerByText(spinnerDistrict, dist)
                editAddressLine.setText(rest)
            }
        } else {
            // 切不到 → 城市/區維持預設，詳細顯示整串，讓使用者自行調整一次
            editAddressLine.setText(addr)
        }
    }

    private fun normalizeCityKey(cityLike: String): String? {
        val want = cityLike.replace("臺", "台")
        return districtsMap.keys.firstOrNull { it.replace("臺", "台") == want }
    }

    // 根據城市（容忍台/臺）綁定地區清單（前面加占位）
    private fun bindDistrictsForCity(cityLike: String) {
        val cityKey = normalizeCityKey(cityLike) ?: return
        val dists = districtsMap[cityKey].orEmpty()
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, listOf(PLACEHOLDER) + dists)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        val current = spinnerDistrict.selectedItem?.toString()
        spinnerDistrict.adapter = adapter

        val keep = dists.indexOfFirst { it == current }
        spinnerDistrict.setSelection(if (keep >= 0) keep + 1 else 0)
    }

    // 由城市 + 完整地址猜測地區名稱（容忍 台/臺）
    private fun guessDistrictForCity(cityLike: String, fullAddress: String): String? {
        val cityKey  = normalizeCityKey(cityLike) ?: return null
        val normAddr = fullAddress.replace("臺", "台")
        val normCity = cityLike.replace("臺", "台")

        // 先把城市前綴拿掉，剩下的字串開頭通常會是地區名
        val rest = if (normAddr.startsWith(normCity)) {
            normAddr.removePrefix(normCity)
        } else {
            normAddr
        }

        val dists = districtsMap[cityKey].orEmpty()
        return dists.firstOrNull { dist ->
            val normDist = dist.replace("臺", "台")
            rest.startsWith(normDist)
        }
    }

    private fun parseAddressByDicts(addr: String): Triple<String, String, String>? {
        if (addr.isBlank()) return null
        val norm = addr.replace("臺", "台")

        // 找城市
        val city = cities.firstOrNull { norm.startsWith(it.replace("臺", "台")) } ?: return null
        val normCity = city.replace("臺", "台")
        var rest = norm.removePrefix(normCity)

        // 找地區（以該城市的清單為準）
        val dists = districtsMap[city].orEmpty()
        val dist = dists.firstOrNull { rest.startsWith(it.replace("臺", "台")) } ?: ""
        val normDist = dist.replace("臺", "台")
        if (dist.isNotBlank()) rest = rest.removePrefix(normDist)

        return Triple(city, dist, rest.trim())
    }

    private fun selectSpinnerByText(spinner: Spinner, text: String) {
        val want = text.replace("臺", "台")
        val adapter = spinner.adapter ?: return
        val count = adapter.count

        if (want.isBlank()) {
            spinner.setSelection(0) // 回到占位
            return
        }

        for (i in 0 until count) {
            val item = adapter.getItem(i)?.toString()?.replace("臺", "台")
            if (item == want) {
                spinner.setSelection(i)
                return
            }
        }
        // 找不到 → 占位
        spinner.setSelection(0)
    }

    // 3) 在 Activity 內任意位置加上這兩個方法
    private fun openLineOfficialAccount(oaId: String) {
        // 深連結（已安裝 LINE）
        val deepLink = Uri.parse("line://ti/p/@$oaId")
        try {
            startActivity(Intent(Intent.ACTION_VIEW, deepLink))
            return
        } catch (_: Exception) {
            // 沒裝 LINE → 走網頁
        }

        // 網頁版「加好友」連結（會導到 LINE 或頁面）
        val webAddFriend = Uri.parse("https://line.me/R/ti/p/@$oaId")
        try {
            startActivity(Intent(Intent.ACTION_VIEW, webAddFriend))
        } catch (_: Exception) {
            // 萬一裝置沒有可處理的瀏覽器，再退到官方頁面
            val profilePage = Uri.parse("https://page.line.me/$oaId")
            try {
                startActivity(Intent(Intent.ACTION_VIEW, profilePage))
            } catch (_: Exception) {
                Toast.makeText(this, "無法開啟 LINE 或瀏覽器", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupAddressSpinners() {
        // 先給地區占位
        spinnerDistrict.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            listOf(PLACEHOLDER)
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }

        // 城市 + 占位
        val cityAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, cities)
        cityAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerCity.adapter = cityAdapter

        // 使用類別層的 bindDistrictsForCity()，不要在這裡再宣告一份
        spinnerCity.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                if (suppressCityListener) return  // 程式化改城市時不動作

                if (pos == 0) {
                    spinnerDistrict.adapter = ArrayAdapter(
                        this@EditProfileActivity,
                        android.R.layout.simple_spinner_item,
                        listOf(PLACEHOLDER)
                    ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
                    spinnerDistrict.setSelection(0)
                    return
                }
                val city = cities[pos]
                bindDistrictsForCity(city) // 類別層版本：會盡量保留原本的地區選項
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        // 初始先停在占位，不多做任何綁定
        spinnerCity.setSelection(0, false)
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
