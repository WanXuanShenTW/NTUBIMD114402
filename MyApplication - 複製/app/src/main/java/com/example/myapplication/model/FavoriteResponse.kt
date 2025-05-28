package com.example.myapplication.model

import com.google.gson.annotations.SerializedName
import com.example.myapplication.model.FavoriteVideo

data class FavoriteResponse(
    @SerializedName("user_id")
    val userId: Int,
    val favorites: List<FavoriteVideo>
)
