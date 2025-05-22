package com.example.myapplication.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.R
import com.example.myapplication.model.VideoEvent

class VideoEventAdapter(
    private var videoEvents: List<VideoEvent>,
    private val onItemClick: (VideoEvent) -> Unit,
    private val onFavoriteClick: (VideoEvent) -> Unit
) : RecyclerView.Adapter<VideoEventAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val txtEventType: TextView = itemView.findViewById(R.id.txtEventType)
        val txtEventTime: TextView = itemView.findViewById(R.id.txtEventTime)
        val txtVideoName: TextView = itemView.findViewById(R.id.txtVideoName)
        val btnFavorite: ImageView = itemView.findViewById(R.id.btnFavorite)

        init {
            itemView.setOnClickListener {
                val position = bindingAdapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onItemClick(videoEvents[position])
                }
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_video_event, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val event = videoEvents[position]
        holder.txtEventType.text = "事件：${event.event_type}"
        holder.txtEventTime.text = "時間：${event.start_time}"
        holder.txtVideoName.text = "影片：${event.video_filename}"

        // 設定愛心圖示
        val heartIcon = if (event.isFavorite) {
            R.drawable.ic_heart_filled
        } else {
            R.drawable.ic_heart_outline
        }
        holder.btnFavorite.setImageResource(heartIcon)

        // 點擊切換收藏狀態
        holder.btnFavorite.setOnClickListener {
            event.isFavorite = !event.isFavorite
            notifyItemChanged(position)
            onFavoriteClick(event) // 回傳目前狀態，讓外部判斷加或取消
        }
    }

    override fun getItemCount() = videoEvents.size

    fun updateData(newList: List<VideoEvent>) {
        videoEvents = newList
        notifyDataSetChanged()
    }
}
