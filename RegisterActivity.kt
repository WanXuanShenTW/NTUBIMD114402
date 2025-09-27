package com.example.myapplication

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import android.content.Intent
import androidx.media3.common.util.UnstableApi
import okhttp3.ResponseBody
import androidx.core.widget.addTextChangedListener
import com.example.myapplication.model.RegisterRequest
import com.example.myapplication.network.RetrofitClient

@UnstableApi
class RegisterActivity : AppCompatActivity() {

    private lateinit var editName: EditText
    private lateinit var editPhone: EditText
    private lateinit var editPassword: EditText
    private lateinit var editConfirmPassword: EditText
    private lateinit var spinnerRole: Spinner
    private lateinit var btnRegister: Button
    private lateinit var backButton: ImageView

    private lateinit var spinnerCity: Spinner
    private lateinit var spinnerDistrict: Spinner
    private lateinit var editAddressLine: EditText
    private lateinit var cityError: TextView
    private lateinit var districtError: TextView
    private lateinit var addressLineError: TextView

    private lateinit var nameError: TextView
    private lateinit var phoneError: TextView
    private lateinit var passwordError: TextView
    private lateinit var confirmPasswordError: TextView

    private lateinit var radioGroupGender: RadioGroup
    private lateinit var genderError: TextView

    private var isPhoneAvailable: Boolean? = null   // null=未知, true=可用, false=已被註冊
    private var phoneCheckInFlight = false

    private fun buildAreas(): LinkedHashMap<String, List<String>> {
        return linkedMapOf(
            "台北市" to listOf("中正區","大同區","中山區","松山區","大安區","萬華區","信義區","士林區","北投區","內湖區","南港區","文山區"),
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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_register)

        editName = findViewById(R.id.editName)
        editPhone = findViewById(R.id.editPhone)
        spinnerRole = findViewById(R.id.spinnerRole)
        editPassword = findViewById(R.id.editPassword)
        editConfirmPassword = findViewById(R.id.editConfirmPassword)
        btnRegister = findViewById(R.id.btnRegister)
        backButton = findViewById(R.id.backButton)

        spinnerCity = findViewById(R.id.spinnerCity)
        spinnerDistrict = findViewById(R.id.spinnerDistrict)
        editAddressLine = findViewById(R.id.editAddressLine)

        nameError = findViewById(R.id.nameError)
        phoneError = findViewById(R.id.phoneError)
        passwordError = findViewById(R.id.passwordError)
        confirmPasswordError = findViewById(R.id.confirmPasswordError)

        radioGroupGender = findViewById(R.id.radioGroupGender)
        genderError = findViewById(R.id.genderError)

        cityError = findViewById(R.id.cityError)
        districtError = findViewById(R.id.districtError)
        addressLineError = findViewById(R.id.addressLineError)

        // 初始化角色下拉選單
        val roleAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, listOf("家屬", "醫護人員"))
        roleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerRole.adapter = roleAdapter

        backButton.setOnClickListener { finish() }

        val areas = buildAreas()
        val cities = listOf("請選擇城市") + areas.keys.toList()
        spinnerCity.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, cities).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        setDistricts(emptyList())

        spinnerCity.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                clearTextView(cityError)
                val selectedCity = cities[position]
                if (position == 0) setDistricts(emptyList())
                else setDistricts(areas[selectedCity].orEmpty())
            }
            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        spinnerDistrict.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                clearTextView(districtError)
            }
            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        setupAutoClearError()

        btnRegister.setOnClickListener {
            val name = editName.text.toString()
            val phone = editPhone.text.toString()
            val password = editPassword.text.toString()
            val confirmPassword = editConfirmPassword.text.toString()
            val city = spinnerCity.selectedItem?.toString().orEmpty()
            val district = spinnerDistrict.selectedItem?.toString().orEmpty()
            val addressLine = editAddressLine.text.toString()

            val gender: String? = when (radioGroupGender.checkedRadioButtonId) {
                R.id.radioFemale -> "F"
                R.id.radioMale -> "M"
                else -> null
            }
            val selectedRoleId = when (spinnerRole.selectedItem.toString()) {
                "家屬" -> 2
                "醫護人員" -> 3
                else -> 2
            }

            // 先清錯誤
            clearError(editName, nameError)
            clearError(editPhone, phoneError)
            clearError(editPassword, passwordError)
            clearError(editConfirmPassword, confirmPasswordError)
            clearError(editAddressLine, addressLineError)
            genderError.visibility = View.GONE
            cityError.visibility = View.GONE
            districtError.visibility = View.GONE

            var hasError = false

            if (name.isEmpty()) {
                setError(editName, nameError, "姓名不能為空"); hasError = true
            }
            if (phone.isEmpty()) {
                setError(editPhone, phoneError, "手機號碼不能為空"); hasError = true
            } else if (!isValidPhone(phone)) {
                setError(editPhone, phoneError, "手機號碼格式錯誤（需為09開頭共10碼）"); hasError =
                    true
            }
            if (password.isEmpty()) {
                setError(editPassword, passwordError, "請輸入密碼"); hasError = true
            } else if (!isValidPassword(password)) {
                setError(
                    editPassword,
                    passwordError,
                    "密碼需至少8碼，含大寫與小寫英文字母"
                ); hasError = true
            }
            if (confirmPassword.isEmpty()) {
                setError(editConfirmPassword, confirmPasswordError, "請再次輸入密碼"); hasError =
                    true
            } else if (password != confirmPassword) {
                setError(editConfirmPassword, confirmPasswordError, "兩次密碼不一致"); hasError =
                    true
            }
            if (radioGroupGender.checkedRadioButtonId == -1) {
                genderError.text = "請選擇性別"
                genderError.visibility = View.VISIBLE
                hasError = true
            }

            // ===== 地址改成「非必填」的關鍵邏輯 =====
            val touchedAddress =
                (city != "請選擇城市") || (district != "請選擇地區") || addressLine.isNotBlank()
            var fullAddress: String? = null
            if (touchedAddress) {
                var addrErr = false
                if (city == "請選擇城市") {
                    cityError.text = "請選擇城市"; cityError.visibility = View.VISIBLE; addrErr =
                        true
                }
                if (district == "請選擇地區" || district.isBlank()) {
                    districtError.text = "請選擇地區"; districtError.visibility =
                        View.VISIBLE; addrErr = true
                }
                if (addressLine.isBlank()) {
                    setError(
                        editAddressLine,
                        addressLineError,
                        "請輸入地址（路名、號、樓層）"
                    ); addrErr = true
                }
                if (addrErr) return@setOnClickListener
                fullAddress = city + district + addressLine
            }
            // =====================================

            if (hasError) return@setOnClickListener

            val req = RegisterRequest(
                name = name,
                phone = phone,
                password = password,
                role_id = selectedRoleId,
                gender = gender!!,
                address = fullAddress   // 可能是 null（完全不填）或完整字串
            )

            RetrofitClient.apiService.registerUser(req)
                .enqueue(object : retrofit2.Callback<ResponseBody> {
                    override fun onResponse(
                        call: retrofit2.Call<ResponseBody>,
                        response: retrofit2.Response<ResponseBody>
                    ) {
                        if (response.isSuccessful) {
                            Toast.makeText(
                                this@RegisterActivity,
                                "註冊成功！請重新登入",
                                Toast.LENGTH_SHORT
                            ).show()
                            startActivity(Intent(this@RegisterActivity, LoginActivity::class.java))
                            finish()
                            return
                        }
                        val raw = response.errorBody()?.string().orEmpty()
                        if (response.code() == 409) {
                            setError(editPhone, phoneError, extractPhoneDuplicateMsg(raw)); return
                        }
                        if (response.code() == 500 && isPhoneDuplicateRaw(raw)) {
                            setError(editPhone, phoneError, "此手機號碼已被註冊"); return
                        }
                        if (response.code() == 422) {
                            try {
                                val obj = org.json.JSONObject(raw)
                                val arr = obj.optJSONArray("detail")
                                if (arr != null) {
                                    var shown = false
                                    for (i in 0 until arr.length()) {
                                        val item = arr.optJSONObject(i) ?: continue
                                        val locArr = item.optJSONArray("loc")
                                        val msg = item.optString("msg", "格式錯誤")
                                        val field =
                                            if (locArr != null && locArr.length() >= 2) locArr.getString(
                                                locArr.length() - 1
                                            ) else ""
                                        if (field == "phone") {
                                            setError(editPhone, phoneError, msg); shown = true
                                        }
                                    }
                                    if (!shown) Toast.makeText(
                                        this@RegisterActivity,
                                        "資料格式錯誤：$raw",
                                        Toast.LENGTH_LONG
                                    ).show()
                                    return
                                }
                            } catch (_: Exception) { /* ignore */
                            }
                        }
                        val err = if (raw.isNotBlank()) raw else "未知錯誤"
                        Toast.makeText(this@RegisterActivity, "註冊失敗：$err", Toast.LENGTH_LONG)
                            .show()
                    }

                    override fun onFailure(call: retrofit2.Call<ResponseBody>, t: Throwable) {
                        Toast.makeText(
                            this@RegisterActivity,
                            "連線失敗：${t.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                })
        }
    }

    private fun clearTextView(tv: TextView) {
        tv.text = ""
        tv.visibility = View.GONE
    }

    private fun setDistricts(districts: List<String>) {
        val list = if (districts.isEmpty()) listOf("請選擇地區") else listOf("請選擇地區") + districts
        spinnerDistrict.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, list).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        districtError.visibility = View.GONE
    }

    private fun setupAutoClearError() {
        editName.addTextChangedListener { clearError(editName, nameError) }
        editPhone.addTextChangedListener { clearError(editPhone, phoneError) }
        editPassword.addTextChangedListener { clearError(editPassword, passwordError) }
        editConfirmPassword.addTextChangedListener { clearError(editConfirmPassword, confirmPasswordError) }
        editAddressLine.addTextChangedListener { clearError(editAddressLine, addressLineError) }
        radioGroupGender.setOnCheckedChangeListener { _, _ -> genderError.visibility = View.GONE }
    }

    private fun isValidPhone(phone: String): Boolean {
        val regex = Regex("^09\\d{8}$")
        return phone.matches(regex)
    }

    private fun isPhoneDuplicateRaw(raw: String): Boolean {
        val lower = raw.lowercase()
        if ("duplicate entry" in lower && "phone" in lower) return true
        val re = Regex("for key '([\\w]+)\\.phone_UNIQUE'", RegexOption.IGNORE_CASE)
        return re.containsMatchIn(raw)
    }

    private fun extractPhoneDuplicateMsg(raw: String): String {
        return try {
            val obj = org.json.JSONObject(raw)
            val field = obj.optString("field", "")
            val msg = obj.optString("message", "")
            if (field.equals("phone", true)) msg.ifBlank { "此手機號碼已被註冊" } else "此手機號碼已被註冊"
        } catch (_: Exception) {
            "此手機號碼已被註冊"
        }
    }

    private fun isValidPassword(password: String): Boolean {
        val regex = Regex("^(?=.*[a-z])(?=.*[A-Z]).{8,}$")
        return password.matches(regex)
    }

    private fun setError(editText: EditText, errorText: TextView, message: String) {
        if (editText.id == R.id.editAddressLine) {
            editText.setBackgroundResource(R.drawable.edittext_underline_error)
        } else {
            editText.setBackgroundResource(R.drawable.edittext_error_background)
        }
        errorText.text = message
        errorText.visibility = View.VISIBLE
    }

    private fun clearError(editText: EditText, errorText: TextView) {
        if (editText.id == R.id.editAddressLine) {
            editText.setBackgroundResource(R.drawable.edittext_underline)
            editText.backgroundTintList = null
        } else {
            editText.setBackgroundResource(R.drawable.edittext_background)
        }
        errorText.visibility = View.GONE
    }
}
