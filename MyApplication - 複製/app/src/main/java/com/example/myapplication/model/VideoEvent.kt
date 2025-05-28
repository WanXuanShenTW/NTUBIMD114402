// VideoEvent.kt
package com.example.myapplication.model

data class VideoEvent(
    var user_id: Int = -1,
    val event_type: String,
    val start_time: String,
    val video_filename: String,
    var isFavorite: Boolean = false
)