import okhttp3.*
import java.io.File
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody


class ApiClient {
    private val client = OkHttpClient()

    fun uploadImage(serverUrl: String, imagePath: String, callback: (String?) -> Unit) {
        val userId = "fixed_user_id"
        val imageFile = File(imagePath)
        if (!imageFile.exists()) {
            callback("Image file does not exist")
            return
        }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("id", userId)
            .addFormDataPart(
                "frame", imageFile.name,
                imageFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url(serverUrl)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                callback(null)
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    callback(response.body?.string())
                } else {
                    callback("Error: ${response.code}")
                }
            }
        })
    }
}
