package com.example.myapplication.adapter

import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication.R
import com.example.myapplication.VideoPlayerActivity
import com.example.myapplication.model.FavoriteVideo

class FavoriteAdapter(
    private val favoriteList: List<FavoriteVideo>
) : RecyclerView.Adapter<FavoriteAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvTitle: TextView = itemView.findViewById(R.id.tvTitle)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_video, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val video = favoriteList[position]
        holder.tvTitle.text = "事件：${video.video_type}，時間：${video.added_at}"

        holder.itemView.setOnClickListener {
            val context = it.context
            val videoUrl = "http://172.20.10.3:5000/videos/${video.video_type}/${video.video_filename}"
            val intent = Intent(context, VideoPlayerActivity::class.java).apply {
                putExtra("video_url", videoUrl)
                putExtra("video_filename", video.video_filename)
            }
            context.startActivity(intent)
        }
    }

    override fun getItemCount(): Int = favoriteList.size
}
