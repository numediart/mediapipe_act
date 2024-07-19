# Mediapipe Facial Landmarks Detection and Animation

## Introduction

This project demonstrates the use of the Mediapipe library to detect facial landmarks in real-time and send blendshapes to a server. The server subsequently transmits the blendshapes to a client, which animates a 3D model based on the received data.

## Installation

You can install the project by cloning the repository:

```bash
git clone https://github.com/numediart/mediapipe_act.git
```

To install the project dependencies, ensure you have Python 3.7 or higher installed. Then, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the project, follow these steps:

1. **Start the Server:**
   Begin by starting the server. The server will handle the reception and distribution of blendshape data.

2. **Launch the Unity Client:**
   Open the Unity client and create a livestream room.

3. **Retrieve and Enter Room ID:**
   In the server console, note the room ID. When prompted by the Mediapipe script, enter this room ID to establish the connection.

By following these steps, the Mediapipe script will successfully send blendshapes to the server, enabling real-time animation of the 3D model in the Unity client.

---

For more information on the Mediapipe library, visit the [official documentation](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=fr).