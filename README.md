# 🛡️ Nirapotta: On-Device ML Safety App

**Nirapotta** (নিরাপত্তা, "safety") is a Flutter + Kotlin Android app that uses an **on-device 1D-CNN (TensorFlow Lite)** to tell genuine emergencies apart from everyday motion, then silently alerts emergency contacts with the user's GPS location. It was built for dense South Asian urban settings, where a visible panic button can put a victim at risk.


> ⚠️ **Experimental prototype.** It can produce false positives and false negatives and is not a replacement for calling emergency services.

---

## 🧠 Machine Learning: The "Brain System"

Cheap sensor thresholds fire on dropped phones, jogging, and loud places. Nirapotta uses a lightweight neural network as a second-stage filter: a simple threshold wakes the system, then the model decides how serious the event is **before any alert is sent**.

### Pipeline

```
Accelerometer (50 Hz) ─┐
                       ├─► Zero-Order Hold sync ─► 10 s circular buffer (500 × 5)
Microphone dB (~2 Hz) ─┘                                  │
                                                          ▼
        Sensor threshold crossed? ──► Min-max normalization ──► 1D-CNN (TFLite)
                                                          │
                                         NONE / MINOR / MAJOR probabilities
                                                          ▼
                                            Fixed-threshold decision gate
```

### Model at a glance

| Item | Details |
|---|---|
| **Task** | 3-class classification of a 10-second sensor window: `NONE`, `MINOR`, `MAJOR` |
| **Input** | 500 timesteps × 5 features (accelerometer motion and microphone sound level, plus a derived motion-magnitude feature) |
| **Model** | 1D Convolutional Neural Network, 27,843 trainable parameters |
| **Deployment** | TensorFlow Lite (`model.tflite`, **39.5 KB**) via `tflite_flutter`, fully offline |
| **Inference latency** | **< 50 ms** on the test device |
| **Evaluation** | Stratified 5-fold cross-validation: **82.7% ± 7.5%** accuracy |

### Architecture

| Layer | Configuration | Output |
|---|---|---|
| Input | | (500, 5) |
| Conv1D | 32 filters, kernel 5, ReLU | (500, 32) |
| MaxPooling1D | pool 2 | (250, 32) |
| Conv1D | 64 filters, kernel 5, ReLU | (250, 64) |
| MaxPooling1D | pool 2 | (125, 64) |
| Conv1D | 64 filters, kernel 3, ReLU | (125, 64) |
| GlobalAveragePooling1D | | (64) |
| Dense | 64 units, ReLU | (64) |
| Dropout | 0.30 | (64) |
| Dense (output) | 3 units, Softmax | (3) |

### Data & Preprocessing

- **Sensor sync:** the accelerometer (50 Hz) and microphone (variable rate) run asynchronously. A **Zero-Order Hold** carries the last dB value forward so every timestep has a complete feature set.
- **Rolling memory:** a fixed-size FIFO circular buffer holds the last 10 seconds (500 readings), overwritten continuously. Raw data never leaves the device.
- **Magnitude feature:** `√(x² + y² + z²)` gives a rotation-invariant measure of movement force.
- **Normalization:** min-max scaling to [0, 1] using training-set statistics stored in `norm-params.json`, so training and inference use identical scaling.
- **Short windows:** if a device delivers fewer than 500 rows (some budget phones sample at 40–48 Hz), the window is zero-padded.

### Dataset

Recordings were collected from a physical Android device in controlled simulations (sample files and collection instructions are in [`Data sample and instructions/`](Data%20sample%20and%20instructions)).

| Class | Label | Scenario | Samples |
|---|---|---|---|
| `NONE` | 0 | Normal phone use: walking, sitting, casual gestures | 25 |
| `SHAKE` → MINOR | 1 | Sustained vigorous shaking without a distress sound | 32 |
| `MAJOR` | 2 | Vigorous shaking **and** loud distress sound together | 30 |

Genuine `MAJOR` events are hard to reproduce, so 6 real recordings were **augmented** with Gaussian noise on the accelerometer, random magnitude scaling (0.85–1.15×), and a regenerated dB column using a synthetic scream envelope (peak 88–100 dB).

### Training

- **Optimizer:** Adam, learning rate 0.001
- **Loss:** sparse categorical cross-entropy
- **Batch size / epochs:** 16 / up to 50
- **Regularization:** dropout (0.30), early stopping (patience 10), learning-rate reduction on plateau
- **Final model:** trained on all 87 samples after cross-validation

| Fold | Accuracy |
|---|---|
| 1 | 83.3% |
| 2 | 83.3% |
| 3 | 82.4% |
| 4 | 94.1% |
| 5 | 70.6% |
| **Mean ± Std** | **82.7% ± 7.5%** |

Variance is high because each validation fold has only 17 samples, so a single error moves a fold by about 5.9 points.

### Decision Logic

| Model output | Decision | Action |
|---|---|---|
| P(MAJOR) ≥ 0.60 | **MAJOR** | Full response: SMS with GPS location to contacts, silent photo capture, audio/video recording |
| P(MINOR) ≥ 0.40 | **MINOR** | Local on-screen warning only, no external communication |
| Otherwise | **NONE** | Event suppressed, monitoring continues |

These thresholds are separate from the home-screen sensitivity sliders, which only control when the sensors wake the Brain System.

### Real-Device Results

Tested on a physical Android 12 phone: **7 of 8 scenarios** gave the expected result.

| Test | Action | Model output (None / Minor / Major) | Decision | Result |
|---|---|---|---|---|
| A1 | Gentle tap | 99.5 / 0.4 / 0.1 | NONE | ✅ |
| A2 | Walking, phone in pocket | 79.8 / 20.2 / 0.0 | NONE | ✅ |
| A3 | Moderate shake, quiet room | 51.5 / 47.5 / 1.0 | MINOR | ✅ |
| A4 | Moderate shake with shouting | 59.2 / 40.8 / 0.0 | MINOR (expected MAJOR) | ❌ |
| B1 | Moderate shake | 0.2 / 95.3 / 4.5 | MINOR | ✅ |
| B2 | Loud clapping only | No inference triggered | NONE | ✅ |
| B3 | Hard shake with loud shouting | 2.9 / 31.8 / 65.3 | MAJOR | ✅ |
| B4 | Phone dropped with a shout | 2.0 / 21.0 / 77.0 | MAJOR | ✅ |

### Known Limitations

- **Small dataset (87 samples, one device).** Results may not generalize to other phones, users, or placements.
- **MAJOR coverage gap.** Training data is mostly high-intensity events, so moderate-intensity emergencies (test A4) can be downgraded to MINOR.
- **Fixed 85 dB sound threshold** can misfire in loud places such as traffic or concerts.
- **Sampling-rate drift** on budget Android devices (40–48 Hz instead of 50 Hz), handled by zero-padding.
- **Android background limits** (Doze mode) can interrupt continuous monitoring.

---

## Features

- **On-device gesture and sound classification** (offline, privacy-preserving)
- **Two-tier alerts:** MINOR (local warning) and MAJOR (SMS with GPS to up to 5 contacts)
- **Silent evidence capture:** photos, audio, and video saved to a PIN-protected Secure Gallery
- **Custom Logic Engine:** map triggers (shake, loud noise, double/triple power-button press) to actions
- **Sensor calibration** and adjustable motion/microphone sensitivity
- **Cancel option:** slide-to-disable on alert screens to avoid accidental alarms

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Flutter (Dart 3.x) |
| Native bridge | Kotlin (MethodChannels, background SMS, power-button receiver) |
| ML | TensorFlow / Keras for training, TensorFlow Lite + `tflite_flutter` for inference |
| Sensors | `sensors_plus`, `noise_meter` |
| Backend / Auth | Firebase (Google Sign-In, Firestore, FCM) |
| Storage | SharedPreferences, CSV sensor logs |


### Usage

1. Accept the safety disclaimer and grant the requested permissions (microphone, camera, location, SMS).
2. Add up to five emergency contacts.
3. Run **Sensor Calibration** to set your baseline.
4. Tap **TAP TO ACTIVATE** to arm monitoring.
5. When an event is detected, the app classifies it and responds as described in [Decision Logic](#decision-logic).

## Troubleshooting

| Problem | Fix |
|---|---|
| `MissingPluginException` | Stop the app completely and run `flutter run` again after adding dependencies. |
| No alert sound | Make sure `assets/alert_sound.mp3` exists and is listed under `assets:` in `pubspec.yaml`. |
| No motion detected | Use a physical device, not an emulator. |
| Alerts not sent in background | Disable battery optimization for the app. |

## Roadmap

- [ ] Larger dataset with moderate-intensity emergencies and more devices and users
- [ ] Adaptive noise floor instead of a fixed 85 dB threshold
- [ ] Deploy Firebase Cloud Functions to enable proximity alerts to nearby users (client side is implemented)
- [ ] SMS retry on failed delivery
- [ ] iOS background support (Swift / Objective-C)

## Author

Shifat Mahmud GitHub: @ShifatMahmud0
