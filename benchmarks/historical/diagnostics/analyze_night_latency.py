import csv

night_videos = ["benchmarks/phase6/621_phase6.csv", "benchmarks/phase6/139_phase6.csv"]

def get_percentile(data, p):
    if not data:
        return 0
    s_data = sorted(data)
    idx = int((len(s_data) - 1) * p / 100.0)
    return s_data[idx]

print("Night Video Latency Distribution (ms)")
for v in night_videos:
    try:
        with open(v, 'r') as f:
            reader = csv.DictReader(f)
            pipeline_ms_list = []
            adapt_ms_list = []
            rois_list = []
            track_ms_list = []
            
            for row in reader:
                try:
                    p_ms = float(row.get('pipeline_ms', row.get('total_ms', 0)))
                    pipeline_ms_list.append(p_ms)
                    
                    if 'adapt_ms' in row:
                        adapt_ms_list.append(float(row['adapt_ms']))
                    if 'n_rois' in row:
                        rois_list.append(float(row['n_rois']))
                    if 'track_ms' in row:
                        track_ms_list.append(float(row['track_ms']))
                except ValueError:
                    pass
            
            print(f"\n--- {v} ---")
            print(f"Frames analyzed: {len(pipeline_ms_list)}")
            
            if pipeline_ms_list:
                print("Pipeline MS Percentiles (50, 90, 95, 99, 100):")
                pcts = [50, 90, 95, 99, 100]
                print(", ".join([f"{p}th: {get_percentile(pipeline_ms_list, p):.2f}" for p in pcts]))
                
            if adapt_ms_list:
                print("CLAHE Adapt MS Percentiles (50, 90, 95, 99, 100):")
                print(", ".join([f"{p}th: {get_percentile(adapt_ms_list, p):.2f}" for p in pcts]))
                
            if rois_list:
                mean_rois = sum(rois_list)/len(rois_list)
                print(f"Re-detection ROIs - Mean: {mean_rois:.2f}, 99th: {get_percentile(rois_list, 99):.2f}")
                
            if track_ms_list:
                mean_track = sum(track_ms_list)/len(track_ms_list)
                print(f"Tracking MS - Mean: {mean_track:.2f}, 99th: {get_percentile(track_ms_list, 99):.2f}")
                
    except Exception as e:
        print(f"Error processing {v}: {e}")
