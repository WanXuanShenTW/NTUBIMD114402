package com.example.myapplication.model

data class FavoriteVideo(
    val record_id: Int,
    val user_id: Int,
    val video_type: String,
    val video_filename: String,
    val pose_before_fall: String?,
    val detected_time: String?,
    val location: String?,
    var is_favorited: Boolean = true
)
