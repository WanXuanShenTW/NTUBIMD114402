package com.example.myapplication.model

data class DeleteEmergencyContactRequest(
    val user_phone: String,
    val contact_phone: String
)
