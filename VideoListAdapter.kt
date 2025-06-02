package com.example.myapplication.adapter

import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.R
import com.example.myapplication.VideoPlayerActivity
import com.example.myapplication.model.VideoEvent
import com.example.myapplication.ApiConfig

@Suppress("UnstableApiUsage")
class VideoListAdapter(
    private val videoList: List<VideoEvent>
) : RecyclerView.Adapter<VideoListAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val txtEventType: TextView = itemView.findViewById(R.id.txtEventType)
        val txtEventTime: TextView = itemView.findViewById(R.id.txtEventTime)
        val txtVideoName: TextView = itemView.findViewById(R.id.txtVideoName)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_video, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val video = videoList[position]

        holder.txtEventType.text = "事件：${video.video_type}"  // ← 改這行
        holder.txtEventTime.text = "時間：${video.detected_time}"
        holder.txtVideoName.text = "影片：${video.video_filename}"

        holder.itemView.setOnClickListener {
            val context = it.context
            val videoUrl = "${ApiConfig.BASE_URL}fall_video_file?record_id=${video.record_id}"
            val intent = Intent(context, VideoPlayerActivity::class.java).apply {
                putExtra("video_url", videoUrl)
                putExtra("video_filename", video.video_filename)
                putExtra("record_id", video.record_id)
            }
            context.startActivity(intent)
        }
    }

    override fun getItemCount(): Int = videoList.size
}
