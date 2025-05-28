package com.example.myapplication.adapter;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.VideoPlayerActivity;
import com.example.myapplication.R;
import com.example.myapplication.model.VideoEvent;

import java.util.List;


public class VideoListAdapter extends RecyclerView.Adapter<VideoListAdapter.ViewHolder> {
    private List<VideoEvent> videoList;
    private Context context;

    public VideoListAdapter(Context context, List<VideoEvent> videoList) {
        this.context = context;
        this.videoList = videoList;
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        public TextView tvTitle;

        public ViewHolder(View itemView) {
            super(itemView);
            tvTitle = itemView.findViewById(R.id.tvTitle);
        }
    }

    @Override
    public VideoListAdapter.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_video, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(VideoListAdapter.ViewHolder holder, int position) {
        VideoEvent videoEvent = videoList.get(position);
        holder.tvTitle.setText(videoEvent.getStart_time());

        // ✅ 加入點擊事件
        holder.itemView.setOnClickListener(v -> {
            Intent intent = new Intent(context, VideoPlayerActivity.class);

            String filename = videoEvent.getVideo_filename();
            String eventType = videoEvent.getEvent_type();
            String videoUrl = "http://172.20.10.3:5000/videos/" + eventType + "/" + filename;

            intent.putExtra("video_url", videoUrl);
            intent.putExtra("video_filename", filename);

            context.startActivity(intent);
        });
    }

    @Override
    public int getItemCount() {
        return videoList.size();
    }
}
