package com.example.myapplication.model

data class VideoEvent(
    val record_id: Int,
    val user_id: Int,
    val video_filename: String,
    val detected_time: String,
    val location: String,
    val pose_before_fall: String,
    var video_type: String = "fall",
    var isFavorite: Boolean = false,
    var in_watchlist: Boolean = false
)
