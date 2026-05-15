# Traffic Analytics

## Project Overview
Traffic Analytics is an automated, AI-powered traffic monitoring system. It processes standard roadway video files to detect, classify, and track vehicles in real-time. Instead of simply overlaying bounding boxes, the system simultaneously generates a dynamic, visual data dashboard alongside the processed video. This transforms raw footage into structured, actionable data regarding traffic density and vehicle distribution.

## Key Features
- **Real-Time Object Detection:** Identifies cars, trucks, buses, and motorcycles with high accuracy.
- **Persistent Object Tracking:** Assigns unique IDs to individual vehicles and maps their driving trajectories across multiple frames.
- **Size-Based Classification:** Calculates the physical footprint of bounding boxes to dynamically categorize vehicles as small, medium, or large.
- **Live Data Dashboard:** Renders a side-by-side analytics panel featuring a live traffic density line graph and a vehicle size distribution bar chart.
- **In-Memory Optimization:** Processes Matplotlib dashboard visuals entirely within system RAM to prevent disk I/O bottlenecks and maximize frame processing speed.

## How It Was Built (Tech Stack)
This project is built entirely in Python, leveraging a stack of industry-standard computer vision and data science libraries:

- **Ultralytics YOLOv8:** Serves as the core neural network for object detection.
- **ByteTrack:** Integrated tracking algorithm used to maintain stable vehicle IDs across sequential frames.
- **OpenCV (cv2):** Handles video ingestion, frame-by-frame matrix manipulation, drawing operations, and final video encoding.
- **Matplotlib & NumPy:** Used in tandem to generate the analytical charts, convert the plot figures into raw RGBA byte buffers, and reshape them into image arrays for OpenCV to stitch into the video output.

## Getting Started

### Prerequisites
- Python 3.8 or higher installed on your system.
- A sample traffic video file named `video.mp4` placed in the root directory of the project.

---

### Option 1: Clone and Run Locally
To set up and run the project directly on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pporalla/Traffic-Analytics.git
   cd Traffic-Analytics
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment:**

   On Windows:
   ```bash
   .\env\Scripts\activate
   ```
   On macOS/Linux:
   ```bash
   source env/bin/activate
   ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the pipeline:**
   ```bash
   python main.py
   ```

Upon completion, the system will output a new file named `output.mp4` containing the tracked video synchronized with the analytics dashboard.

---

### Option 2: Fork the Repository
If you wish to contribute or modify the project for your own use:

1. Click the **Fork** button at the top right of this repository's GitHub page.
2. Clone your forked copy to your local machine using the instructions in Option 1.
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
4. Commit your changes and push the branch to your fork.
5. Open a **Pull Request** if you wish to merge changes back to the main project.

---

## Project Structure
```text
Traffic-Analytics/
├── main.py              # Main application script and processing loop
├── requirements.txt     # List of required Python dependencies
├── .gitignore           # Git ignore file for environment and media files
└── README.md            # Project documentation
```

## Acknowledgments and Credits
This project relies heavily on the incredible work of the open-source community. Special thanks to the teams behind the following packages:

- **Ultralytics (YOLO):** https://github.com/ultralytics/ultralytics
- **OpenCV Python:** https://github.com/opencv/opencv-python
- **Matplotlib:** https://github.com/matplotlib/matplotlib
- **NumPy:** https://github.com/numpy/numpy
