import os
import argparse
import time
from faster_whisper import WhisperModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe lectures using Faster-Whisper.")

    # Paths
    parser.add_argument("--input_dir", default="audios", help="Folder containing audio files")
    parser.add_argument("--output_dir", default="transcripts", help="Folder to save text files")
    
    # Model Settings
    parser.add_argument("--model", default="medium", help="tiny, base, small, medium, large-v2")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (use cuda if you have a GPU)")
    parser.add_argument("--compute_type", default="int8", help="int8, float16")
    
    # Tuning
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size (1-10). Higher = more accurate but slower.")
    
    args = parser.parse_args()

    if args.device == "cpu" and args.beam_size > 3:
        print(f"\n⚠️  WARNING: You are using device='cpu' with a beam_size of {args.beam_size}.")
        print("   This may be slow. Consider using --beam_size 1 or --beam_size 2 for faster CPU performance.\n")

    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔹 Loading {args.model} model on {args.device}...")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    valid_ext = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_ext)]

    if not files:
        print("⚠️ No audio files found.")
        return

    print(f"🔹 Found {len(files)} lectures. Starting transcription...\n")

    for i, file in enumerate(files, 1):
        file_path = os.path.join(args.input_dir, file)
        out_path = os.path.join(args.output_dir, os.path.splitext(file)[0] + ".txt")

        print(f"[{i}/{len(files)}] 🎙️  Transcribing: {file}")
        start_time = time.time()

        try:
            segments, info = model.transcribe(
                file_path, 
                beam_size=args.beam_size, 
                vad_filter=True 
            )

            print(f"   Language: {info.language.upper()} | Duration: {info.duration/60:.2f} min")

            #Open file to write result
            with open(out_path, "w", encoding="utf-8") as f:
                previous_end = 0
                
                with tqdm(total=info.duration, unit="s", bar_format="{l_bar}{bar}| {n:.0f}/{total_fmt}s") as pbar:
                    for segment in segments:
                        #Paragraph Logic
                        gap = segment.start - previous_end
                        if gap > 2.0:
                            f.write("\n\n")
                        
                        f.write(segment.text.strip() + " ")
                        
                        previous_end = segment.end
                        pbar.update(segment.end - pbar.n)

            elapsed = time.time() - start_time
            print(f"   ✅ Saved to: {out_path} (Took {elapsed:.1f}s)\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("🎉 All lectures transcribed!")

if __name__ == "__main__":
    main()