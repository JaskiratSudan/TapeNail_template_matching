# TapeNail: Secure Your Phone Access Through Your Nail

**TapeNail** is a hardware-software authentication system that uses passive 3D optical patterns embedded in nail art to unlock smartphones. The system leverages polarized light, real-time camera capture, and a lightweight deep learning model to detect and verify unique, user-defined optical signatures without relying on permanent biometrics.

---

## Repository Structure

| Folder                     | Description                     |
|---------------------------|---------------------------------|
<<<<<<< HEAD
| `ANDROID_APP/`            | Mobile app code |
| `SCRIPTS/`       | Prototype code  |
| `DOCUMENTS/` | Docs, drafts, figures, notes     |
=======
| `Android_App/`            | Mobile app code |
| `Scripts/`       | Prototype code  |
| `DOCUMENTS/` | Docs, drafts, figures, notes     |
| `TAPENAIL_test/`          | App-yolo integration test    |
>>>>>>> 88a153f1e47e6540576475d210697ba7e1b497e8
| `TAPENAIL_YOLO/`          | YOLO-based pattern detection    |

---

## Features

- Revocable authentication with passive optical tokens
- Personalized pattern design (3D + layered aesthetics)
- Secure against spoofing, cloning, and replay attacks
- On-device YOLOv11-n model (lightweight, fast)
- No additional hardware — works with phone camera

---

## Tooling

- **Android**: Java / Kotlin (Camera2 API, TFLite)
- **Model Training**: YOLOv11-n using Ultralytics
- **Dataset Management**: Roboflow + manual augmentation
- **Pattern Design**: Transparent tape, polarized films, glitter base

---

## Getting Started

Clone the repository and navigate into the Android app:

```bash
git clone https://github.com/your-org/TapeNail.git
<<<<<<< HEAD
cd TapeNail/ANDROID_APP
=======
cd TapeNail/Android_App
>>>>>>> 88a153f1e47e6540576475d210697ba7e1b497e8



