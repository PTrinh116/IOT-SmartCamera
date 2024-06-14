# [NT532.O21]
# Smart Camera Project

## Overview

Overview
This project implements a smart camera system using Raspberry Pi 4. The system utilizes a Python script camera.py to capture video footage. Prior to recording, a PIR (Passive Infrared) sensor detects motion to activate the camera. When the script detects a human face in the captured footage, it transmits the data to a server for facial recognition processing. The server then analyzes the data and returns the recognized results.

## Requirements

- Raspberry Pi 4
- Camera module compatible with Raspberry Pi
- Internet connection
- Server for facial recognition (e.g., AWS, Azure, or a custom server)

## Usage

### Run the Camera Script

Execute `camera.py` on your Raspberry Pi to start capturing video and detecting faces.
python camera.py
Server Processing: The server will receive the video stream from the Raspberry Pi. Upon detecting a face, it will perform facial recognition and send back the results.
