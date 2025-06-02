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
    private var videoEvents: MutableList<VideoEvent>,
    private val onItemClick: (VideoEvent) -> Unit,
    private val onFavoriteClick: (VideoEvent, Int, () -> Unit) -> Unit
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
        holder.txtEventType.text = "事件：${event.video_type}"   // ← 改這行
        holder.txtEventTime.text = "時間：${event.detected_time}"
        holder.txtVideoName.text = "影片：${event.video_filename}"

        val heartIcon = if (event.isFavorite) {
            R.drawable.ic_collect_filled
        } else {
            R.drawable.ic_collect_outline
        }
        holder.btnFavorite.setImageResource(heartIcon)

        holder.btnFavorite.setOnClickListener {
            val pos = holder.bindingAdapterPosition
            if (pos != RecyclerView.NO_POSITION) {
                val targetEvent = videoEvents[pos]
                onFavoriteClick(targetEvent, pos) {
                    // API 成功後由外部呼叫 done() → 更新愛心狀態
                    notifyItemChanged(pos)
                }
            }
        }
    }

    override fun getItemCount() = videoEvents.size

    fun updateFavoriteByRecordId(recordId: Int, isFavorite: Boolean) {
        videoEvents.forEachIndexed { index, event ->
            if (event.record_id == recordId) {
                event.isFavorite = isFavorite
                notifyItemChanged(index)
            }
        }
    }

    fun updateData(newList: List<VideoEvent>) {
        videoEvents = newList.toMutableList()
        notifyDataSetChanged()
    }

    fun getItemPosition(event: VideoEvent): Int {
        return videoEvents.indexOfFirst {
            it.video_filename == event.video_filename &&
                    it.video_type == event.video_type &&
                    it.detected_time == event.detected_time
        }
    }
}
