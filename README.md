# Real-Time Industrial Bottling Inspection & Tracking System

An automated computer vision and deep learning pipeline designed for high-speed industrial quality control. This system detects unsealed/uncapped water bottles on conveyor belts under dense occlusion and high-throughput scenarios.

---

## 📌 Key Features & Architecture

* **Zero-Shot Object Segmentation:** Integrated **Grounded-SAM** for precise visual feature segmentation and automated dataset annotation of tiny defects (missing caps).
* **Real-Time Detection:** Deployed **YOLO** for high-speed inference on bottle body (Bottle) and cap (Cap) detection.
* **High-Density Object Tracking:** Implemented **Bot-SORT** to handle dense spatial occlusion, maintaining persistent track IDs across crowded conveyor arrangements without ID switching.
* **Hardware Interfacing Concept:** Designed an architectural extension to output rejection signals via **PLC protocols (Modbus TCP / OPC UA)** to actuate mechanical ejectors in real-time.

---

## 🛠️ Tech Stack & Frameworks

* **Languages:** Python, C++
* **AI & Computer Vision Frameworks:** PyTorch, OpenCV, YOLO, Grounded-SAM, Bot-SORT
* **CAD & Mechanical Design:** SolidWorks
* **Industrial Automation Concept:** PLC Integration (Modbus / OPC UA)

---

## 🔮 Future Roadmap: Industrial PLC Integration

1. **Edge Deployment:** Optimizing PyTorch model weights via TensorRT for low-latency edge devices.
2. **Hardware Handshake:** Direct I/O triggering from inference script to industrial PLC controllers.
