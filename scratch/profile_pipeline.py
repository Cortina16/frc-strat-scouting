import asyncio
import time
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.actors import call_actor

async def main():
    orchestrator = PipelineOrchestrator(
        use_ray=False,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_mock_mode=True,
        parquet_export_dir="data/parquet",
        config_path="configs/pipeline.yaml",
        detector_engine_path="data/models/einstein_raw_v1.pt",
        red_teams=[1323, 4065, 4414],
        blue_teams=[868, 2046, 2910],
    )
    await orchestrator.start_pipeline("output.mp4", "bench_test")
    actors = orchestrator.actors
    
    for b_idx in range(5):
        t0 = time.perf_counter()
        batch = await call_actor(actors.ingestion, "get_next_gameplay_batch", 8)
        t_ing = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        estimates, det_results, per_frame_ocr = await orchestrator._process_vision_batch(batch)
        t_vis = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        frame_indices = [f.metadata.frame_index for f in batch]
        detections_batch = [det.boxes for det in det_results]
        tracks_batch = await call_actor(
            actors.tracking,
            "update_tracks_batch",
            frame_indices,
            detections_batch,
            per_frame_ocr,
            estimates,
            batch,
            is_autonomous=False,
        )
        t_trk = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        for i, frame in enumerate(batch):
            h_est = estimates[i] if i < len(estimates) else None
            r_tracks = tracks_batch[i] if i < len(tracks_batch) else []
            r_boxes = [t.pixel_bbox for t in r_tracks if t.pixel_bbox]
            dual_res = orchestrator._ball_detector.detect_dual_zone(frame.image_bgr, robot_boxes=r_boxes)
            orchestrator._ball_tracker.update(
                detections=dual_res.individual_balls,
                frame_index=frame.metadata.frame_index,
                homography=h_est,
                robot_boxes=r_boxes,
                clusters=dual_res.clusters,
            )
        t_ball = time.perf_counter() - t0
        
        t_total = t_ing + t_vis + t_trk + t_ball
        print(f"Batch {b_idx}: frames {batch[0].metadata.frame_index}-{batch[-1].metadata.frame_index} | Total: {t_total*1000:.1f}ms ({8/t_total:.1f} FPS) | Ingest: {t_ing*1000:.1f}ms, Vision: {t_vis*1000:.1f}ms, Trk: {t_trk*1000:.1f}ms, Ball: {t_ball*1000:.1f}ms")

    await orchestrator.stop_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
