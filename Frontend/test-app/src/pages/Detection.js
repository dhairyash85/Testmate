import React, { useRef, useEffect } from "react";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs";

const VideoFaceDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const setupCamera = async () => {
      const video = videoRef.current;
      if (!video) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        video.srcObject = stream;
        await video.play();
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    const detectFace = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) return;

      const model = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
          maxFaces: 1,
          refineLandmarks: true,
        }
      );

      const ctx = canvas.getContext("2d");

      const renderLoop = async () => {
        if (video.readyState === 4) {
          const faces = await model.estimateFaces(video);

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          if (faces.length > 0) {
            faces.forEach((face) => {
              ctx.beginPath();
              ctx.strokeStyle = "red";
              face.keypoints.forEach((keypoint) => {
                const { x, y } = keypoint;
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
              });
              ctx.stroke();
            });
          }
        }
        requestAnimationFrame(renderLoop);
      };

      renderLoop();
    };

    setupCamera().then(() => {
      videoRef.current.onloadeddata = detectFace;
    });
  }, []);

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{ border: "1px solid black" }}
      />
    </div>
  );
};

export default VideoFaceDetector;
