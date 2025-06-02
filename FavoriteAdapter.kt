package com.example.myapplication.adapter

import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.R
import com.example.myapplication.VideoPlayerActivity
import com.example.myapplication.model.FavoriteVideo
import com.example.myapplication.ApiConfig

@androidx.media3.common.util.UnstableApi
class FavoriteAdapter(
    private val favoriteList: MutableList<FavoriteVideo>,
    private val onRemoveClick: (FavoriteVideo, Int, () -> Unit) -> Unit,
    private val onAddClick: (FavoriteVideo, Int, () -> Unit) -> Unit
) : RecyclerView.Adapter<FavoriteAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val txtEventType: TextView = itemView.findViewById(R.id.txtEventType)
        val txtEventTime: TextView = itemView.findViewById(R.id.txtEventTime)
        val txtVideoName: TextView = itemView.findViewById(R.id.txtVideoName)
        val btnFavorite: ImageView = itemView.findViewById(R.id.btnRemoveFavorite) // 可以改名更語意化
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_video, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val video = favoriteList[position]

        holder.txtEventType.text = "事件：${video.pose_before_fall ?: "未知"}"
        holder.txtEventTime.text = "時間：${video.detected_time ?: "無資料"}"
        holder.txtVideoName.text = "地點：${video.location ?: "未知"}\n影片：${video.video_filename}"

        val iconRes = if (video.is_favorited) {
            R.drawable.ic_collect_filled
        } else {
            R.drawable.ic_collect_outline
        }
        holder.btnFavorite.setImageResource(iconRes)

        // 播放影片
        holder.itemView.setOnClickListener {
            val context = it.context
            // val videoUrl = "https://d64d-150-116-79-4.ngrok-free.app/fall_video_file?record_id=${video.record_id}"
            val videoUrl = "${ApiConfig.BASE_URL}fall_video_file?record_id=${video.record_id}"
            val intent = Intent(context, VideoPlayerActivity::class.java).apply {
                putExtra("video_url", videoUrl)
                putExtra("video_filename", video.video_filename)
                putExtra("record_id", video.record_id)
            }
            context.startActivity(intent)
        }

        // 點擊愛心：雙向收藏/取消
        holder.btnFavorite.setOnClickListener {
            val realPosition = holder.adapterPosition
            if (realPosition != RecyclerView.NO_POSITION) {
                val targetVideo = favoriteList[realPosition]

                if (targetVideo.is_favorited) {
                    onRemoveClick(targetVideo, realPosition) {
                        targetVideo.is_favorited = false
                        notifyItemChanged(realPosition)
                    }
                } else {
                    onAddClick(targetVideo, realPosition) {
                        targetVideo.is_favorited = true
                        notifyItemChanged(realPosition)
                    }
                }
            }
        }
    }

    override fun getItemCount(): Int = favoriteList.size
}
