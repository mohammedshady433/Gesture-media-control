# Hand Gesture Media Control - Simplified

A single-file hand gesture recognition system for controlling media playback.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_simple.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run gesture_app.py
   ```

## 🎮 Gestures

| Gesture | Action |
|---------|--------|
| 👆 Thumbs Up | Volume Up |
| 👎 Thumbs Down | Volume Down |
| ✌️ Peace Sign | Play/Pause |
| ☝️ Pointing | Next Track |
| ✋ Open Palm | Previous Track |
| ✊ Fist | Mute |
| 🤏 Pinch | Volume Control (drag) |

## 📁 Simplified Structure

```
Project/
├── gesture_app.py           # Single file with all functionality
├── requirements_simple.txt  # Minimal dependencies
└── README_SIMPLE.md        # This file
```

## ⚙️ Configuration

Edit constants at the top of `gesture_app.py`:

```python
MIN_DETECTION_CONFIDENCE = 0.8  # Hand detection sensitivity
MIN_TRACKING_CONFIDENCE = 0.7   # Tracking smoothness
PINCH_THRESHOLD = 0.06          # Pinch detection
FRAME_WIDTH = 1280              # Camera resolution
FRAME_HEIGHT = 720
```

## 💡 Tips

- Use good lighting
- Keep hand 40-60cm from camera
- Make clear, deliberate gestures
- Hold gesture for 0.5-1 second

---

**Made simple for easy understanding and customization! 🎉**
