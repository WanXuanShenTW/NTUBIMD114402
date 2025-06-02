package com.example.myapplication.network

import retrofit2.Call
import retrofit2.http.*
import com.example.myapplication.model.FavoriteResponse
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.model.FavoriteRequest
import com.example.myapplication.model.FavoriteVideo


interface ApiService {

    @POST("watchlist")
    fun addFavorite(@Body request: FavoriteRequest): Call<Map<String, String>>

    @HTTP(method = "DELETE", path = "watchlist", hasBody = true)
    fun removeFavorite(@Body request: FavoriteRequest): Call<Map<String, String>>

    @GET("watchlist")
    fun getFavorites(
        @Query("user_id") userId: Int,
        @Query("video_type") videoType: String = "fall"
    ): Call<List<FavoriteVideo>>


    @GET("fall_videos_data")
    fun getFallVideos(
        @Query("caregiver_id") caregiverId: Int,
        @Query("elder_id") elderId: Int,
        @Query("start_date") startDate: String,
        @Query("end_date") endDate: String,
        @Query("limit") limit: Int = 5
    ): Call<List<VideoEvent>>
}
