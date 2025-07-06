package com.example.myapplication.model

data class EmergencyContactRequest(
    val user_phone: String,
    val contact_phone: String,
    val priority: Int? = null,
    val relationship: String? = null
)
