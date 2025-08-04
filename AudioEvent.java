package be.tarsos.dsp;

public class AudioEvent {
	private final float[] floatBuffer;
	private final int bufferSize;

	public AudioEvent(float[] floatBuffer, int bufferSize) {
		this.floatBuffer = floatBuffer;
		this.bufferSize = bufferSize;
	}

	public float[] getFloatBuffer() {
		return floatBuffer;
	}

	public int getBufferSize() {
		return bufferSize;
	}
}
