package com.example.myapplication.model

data class UpdateUserRequest(
    val phone: String,
    val name: String?,
    val oldPassword: String? = null,
    val password: String?
)
