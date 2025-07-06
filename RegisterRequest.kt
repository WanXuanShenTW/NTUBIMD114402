package com.example.myapplication.model

data class RegisterRequest(
    val name: String,
    val phone: String,
    val password: String,
    val role_id: Int
)
