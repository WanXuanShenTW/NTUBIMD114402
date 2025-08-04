package be.tarsos.dsp;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class AudioDispatcher {

	private final UniversalAudioInputStream audioStream;
	private final int bufferSize;
	private final int overlap;
	private final List<AudioProcessor> processors;
	private boolean stopped;

	public AudioDispatcher(InputStream stream, int bufferSize, int overlap) {
		this.audioStream = new UniversalAudioInputStream(stream);
		this.bufferSize = bufferSize;
		this.overlap = overlap;
		this.processors = new LinkedList<>();
		this.stopped = false;
	}

	public void addAudioProcessor(AudioProcessor processor) {
		processors.add(processor);
	}

	public void run() {
		int stepSize = bufferSize - overlap;
		int totalBytes = bufferSize * 2; // 16-bit mono audio

		byte[] buffer = new byte[totalBytes];
		float[] floatBuffer = new float[bufferSize];

		try {
			while (!stopped && audioStream.read(buffer, 0, totalBytes) != -1) {
				// 轉成 float [-1.0, 1.0]
				for (int i = 0; i < bufferSize; i++) {
					int low = buffer[2 * i] & 0xff;
					int high = buffer[2 * i + 1];
					int sample = (high << 8) | low;
					floatBuffer[i] = sample / 32768f;
				}

				AudioEvent event = new AudioEvent(floatBuffer, bufferSize);
				for (AudioProcessor processor : processors) {
					if (!processor.process(event)) break;
				}
			}
		} catch (IOException e) {
			throw new RuntimeException("讀取音訊資料失敗", e);
		}
	}

	public void stop() {
		this.stopped = true;
	}

	// ✅ 封裝內部專用的 UniversalAudioInputStream，不佔外部命名空間
	private static class UniversalAudioInputStream extends InputStream {
		private final InputStream stream;

		public UniversalAudioInputStream(InputStream stream) {
			this.stream = stream;
		}

		@Override
		public int read(byte[] b, int off, int len) throws IOException {
			return stream.read(b, off, len);
		}

		@Override
		public int read() throws IOException {
			return stream.read();
		}

		@Override
		public void close() throws IOException {
			stream.close();
		}
	}
}
