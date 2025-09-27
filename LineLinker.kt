package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.net.Uri

object LineLinker {
    private const val LINE_PACKAGE = "jp.naver.line.android"
    private const val OA_ID_NO_AT = "883lincr"      // ← 改成你的（不含 @）
    private const val OA_ID_WITH_AT = "@883lincr"   // ← 改成你的（含 @）

    /** 一鍵加入好友（已加則開聊天室） */
    fun openAddFriend(context: Context) {
        val deep = "line://ti/p/$OA_ID_WITH_AT"
        val web  = "https://line.me/R/ti/p/$OA_ID_WITH_AT"
        openWithLineOrBrowser(context, deep, web)
    }

    /** 開啟 OA 首頁（官網頁） */
    fun openOfficialPage(context: Context) {
        val web = "https://page.line.me/$OA_ID_NO_AT"
        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(web)))
    }

    /** （可選）開啟你的 LIFF 綁定頁 */
    fun openLiffBind(context: Context, liffId: String, query: String = "") {
        val deep = "line://app/$liffId${if (query.isNotEmpty()) "?$query" else ""}"
        val web  = "https://liff.line.me/$liffId${if (query.isNotEmpty()) "?$query" else ""}"
        openWithLineOrBrowser(context, deep, web)
    }

    private fun openWithLineOrBrowser(context: Context, deepLink: String, webLink: String) {
        try {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(deepLink)).apply {
                setPackage(LINE_PACKAGE)
            })
        } catch (_: Exception) {
            try {
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(webLink)))
            } catch (_: Exception) {
                context.startActivity(
                    Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=$LINE_PACKAGE"))
                )
            }
        }
    }
}
