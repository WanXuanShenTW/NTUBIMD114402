package com.example.myapplication

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class VoiceResultActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_result)

        val textView = findViewById<TextView>(R.id.textViewVoiceResult)
        val result = getSharedPreferences("smartcare_pref", MODE_PRIVATE)
            .getString("latest_voice_result", "尚無語音紀錄")

        textView.text = result
    }
}
