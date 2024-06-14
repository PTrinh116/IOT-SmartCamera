# [NT532.O21]
# Smart Camera Project

## Overview

This project implements a smart camera system using Raspberry Pi 4. The system runs a Python script `camera.py` that captures video footage. When the script detects a human face, it sends the footage to a server for further processing. The server is responsible for facial recognition and returns the results.

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
