package be.tarsos.dsp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class AudioDispatcherFactory {

    public static AudioDispatcher fromPipe(String audioFilePath, int sampleRate, int audioBufferSize, int bufferOverlap) {
        try {
            File file = new File(audioFilePath);
            FileInputStream inputStream = new FileInputStream(file);

            // 不要用 UniversalAudioInputStream（這個類別不存在了）
            // AudioDispatcher 內部會自動包裝 InputStream
            return new AudioDispatcher(inputStream, audioBufferSize, bufferOverlap);

        } catch (IOException e) {
            throw new RuntimeException("無法讀取音訊檔案: " + audioFilePath, e);
        }
    }
}