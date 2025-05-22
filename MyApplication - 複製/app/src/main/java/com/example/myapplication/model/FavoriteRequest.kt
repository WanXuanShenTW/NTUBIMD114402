package com.example.myapplication.model

data class FavoriteRequest(
    val user_id: Int,
    val video_filename: String,
    val video_type: String
)
