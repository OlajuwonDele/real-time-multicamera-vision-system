# Real-Time Multi-Camera Video Inference & Tracking Pipeline

**Project Overview**  
Modern robotics and AI systems often require real-time processing of multiple video streams for object detection, tracking, and decision-making. This project demonstrates a complete end-to-end pipeline capable of ingesting single or multiple video streams, performing real-time inference and tracking, and supporting scalable deployment.

---

## 🚀 Features

- **Multi-Camera Support** – Ingest video from webcams, RTSP streams, or video files.  
- **Object Detection & Tracking** – Built-in support for YOLOv8 detection and modular trackers like SORT, DeepSORT, or ByteTrack.  
- **Modular Inference Backends** – Supports PyTorch, ONNX Runtime, and TensorRT for optimized GPU/CPU performance.  
- **Distributed Pipeline Ready** – Designed for multi-process or multi-node deployment for scaling.  
- **Performance Benchmarking** – Measures FPS, latency, and accuracy across cameras and backends.  
- **Config-Driven** – YAML-based configuration for easy swapping of sources, models, and runtime parameters.  
- **Optional Web-Based UI** – Visualize streams, detections, and tracking in real-time.  
- **Dockerized Deployment** – Fully containerized for reproducibility and portability.
