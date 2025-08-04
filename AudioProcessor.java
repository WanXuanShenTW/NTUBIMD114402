package be.tarsos.dsp;

public interface AudioProcessor {
	boolean process(AudioEvent audioEvent);
	void processingFinished();
}
