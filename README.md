# Motion Detection Email Alert System

This project is a real-time motion detection application that monitors a webcam feed and sends an email alert when movement is detected.

## Overview
The system uses background frame comparison to detect motion, captures the most relevant frame, and sends it via email using SMTP. Cleanup and email operations are handled asynchronously for performance.

## Tech Stack
- Python
- OpenCV
- SMTP (Gmail)
- Multithreading
- python-dotenv

## Features
- Real-time motion detection using a webcam
- Automatic image capture on movement
- Email alerts with image attachments
- Background folder cleanup
- Multithreaded execution for responsiveness

## Files
- `main.py` – Motion detection logic and video processing
- `emailing.py` – Email alert functionality
- `images/` – Temporary storage for captured frames

## Usage
1. Install dependencies
2. Set environment variables
3. Run


## Learning Outcome
This project demonstrates practical computer vision, event-driven automation, and integrating real-time systems with email notifications.
