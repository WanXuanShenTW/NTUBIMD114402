package com.example.myapplication.network

import retrofit2.Call
import retrofit2.http.*
import com.example.myapplication.model.FavoriteResponse
import com.example.myapplication.model.VideoDateResponse
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.model.FavoriteRequest


interface ApiService {

    @GET("video-date/{video_filename}")
    fun getVideoDate(@Path("video_filename") filename: String): Call<VideoDateResponse>

    @POST("/add_favorite")
    fun addFavorite(@Body data: FavoriteRequest): Call<Map<String, String>>

    @POST("/remove_favorite")
    fun removeFavorite(@Body data: FavoriteRequest): Call<Map<String, String>>

    @GET("favorites/{user_id}")
    fun getFavorites(@Path("user_id") userId: Int): Call<FavoriteResponse>

    @GET("video-events")
    fun getVideoEvents(
        @Query("user_id") userId: Int,
        @Query("start_date") startDate: String,
        @Query("end_date") endDate: String
    ): Call<List<VideoEvent>>
}
