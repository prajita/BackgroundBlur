import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";
const { ImageSegmenter, SegmentationMask, FilesetResolver } = vision;
const videoElement = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const blurBtn = document.getElementById('blur-btn');
const unblurBtn = document.getElementById('unblur-btn');
const outputCanvasCtx = canvas.getContext('2d');
const fpsTextElement = document.getElementById("fpsText");
const bgCanvas = document.getElementById('canvas');
const bgCanvasCtx = bgCanvas.getContext("2d");
// Define the range of confidence score for background blur effect.
let minConfidence = 0.4;
let imageSegmenter;
let runningMode = "VIDEO";

startBtn.addEventListener('click', e => {
  startBtn.disabled = true;
  stopBtn.disabled = false;

  unblurBtn.disabled = false;
  blurBtn.disabled = false;

  startVideoStream();
});

stopBtn.addEventListener('click', e => {
  startBtn.disabled = false;
  stopBtn.disabled = true;

  unblurBtn.disabled = true;
  blurBtn.disabled = true;

  unblurBtn.hidden = true;
  blurBtn.hidden = false;

  videoElement.hidden = false;
  canvas.hidden = true;

  stopVideoStream();
});

blurBtn.addEventListener('click', e => {
  blurBtn.hidden = true;
  unblurBtn.hidden = false;

  videoElement.hidden = true;
  canvas.hidden = false;

  loadMediaPipeImageSegmenter();
});

unblurBtn.addEventListener('click', e => {
  blurBtn.hidden = false;
  unblurBtn.hidden = true;

  videoElement.hidden = false;
  canvas.hidden = true;
});

videoElement.onplaying = () => {
  canvas.height = videoElement.videoHeight;
  canvas.width = videoElement.videoWidth;
};

function startVideoStream() {
  navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 480 },
      height: { ideal: 320 },
    },
    audio: false
  })
    .then(stream => {
      videoElement.srcObject = stream;
      videoElement.play();
    })
    .catch(err => {
      startBtn.disabled = false;
      blurBtn.disabled = true;
      stopBtn.disabled = true;
      alert(`Following error occurred: ${err}`);
    });
}


function stopVideoStream() {
  const stream = videoElement.srcObject;

  stream.getTracks().forEach(track => track.stop());
  videoElement.srcObject = null;
}

async function loadMediaPipeImageSegmenter() {
  const modelPath = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
  try {
    imageSegmenter = await createImageSegmenter(modelPath);
    if (hasGetUserMedia()) {
      enableCamera();
    } else {
      alert("getUserMedia() is not supported by your browser");
    }
  } catch (error) {
    console.error("Error loading the MediaPipe Image Segmenter:", error);
  }
}
// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function createImageSegmenter(modelPath) {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    modelPath
  );

  imageSegmenter = await ImageSegmenter.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
      delegate: "GPU"
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: true
  });

  return imageSegmenter;
}


// Get segmentation from the webcam
let isFirstFrame = true;
async function predictWebcam() {
  if (isFirstFrame) {
    isFirstFrame = false;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  outputCanvasCtx.save();
  outputCanvasCtx.clearRect(0, 0, video.videoWidth, video.videoHeight);
  outputCanvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  // Do not run ML model if imageSegmenter hasn't loaded.
  if (imageSegmenter === undefined) {
    return;
  }

  // Start segmenting the stream.
  const nowInMs = Date.now();
  imageSegmenter.segmentForVideo(canvas, nowInMs, drawSegmentationResult);
}
// Enable the live webcam view and start imageSegmentation.
async function enableCamera() {
  if (imageSegmenter === undefined) {
    return;
  }
  // getUsermedia parameters.
  const constraints = {
    video: {
      width: { ideal: 480 },
      height: { ideal: 320 },
    },
  };

  // Activate the webcam stream.
  video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  video.addEventListener("loadeddata", predictWebcam);
}

// Callback get executed for every frame that was segmented.
let lastFrameMs = Date.now();
function drawSegmentationResult(result) {
  const now = Date.now();
  const timeSinceLastFrame = now - lastFrameMs;
  lastFrameMs = now;
  fpsTextElement.innerHTML = "FPS: " + Math.round(1000 / timeSinceLastFrame);

  const confidenceMasks = result.confidenceMasks[0].getAsFloat32Array();

  // Draw the webcam frame on the output canvas.
  outputCanvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Create an ImageData object from the webcam frame.
  const webcamImageData = outputCanvasCtx.getImageData(0, 0, canvas.width, canvas.height);

  // Draw the background blur effect by applying the blur to the background.
  const backgroundBlurImageData = applyBackgroundBlur(webcamImageData, confidenceMasks);

  // Combine the segmented video with the background blur.
  const combinedImageData = combineImages(webcamImageData, backgroundBlurImageData, confidenceMasks);

  // Put the resulting image back on the output canvas.
  outputCanvasCtx.putImageData(combinedImageData, 0, 0);

  // Continue processing frames.
  window.requestAnimationFrame(predictWebcam);
}

// Combine two ImageData objects.
function combineImages(imageData1, imageData2, confidenceMasks) {
  const width = canvas.width;
  const height = canvas.height;

  for (let i = 0; i < width * height * 4; i += 4) {
    // If the pixel in the segmentation mask is confident, use the segmented video pixel.
    if (confidenceMasks[i / 4] >= minConfidence) {
      imageData1.data[i] = imageData2.data[i];
      imageData1.data[i + 1] = imageData2.data[i + 1];
      imageData1.data[i + 2] = imageData2.data[i + 2];
    }
  }

  return imageData1;
}

// Apply a background blur effect to the image.
function applyBackgroundBlur(imageData, confidenceMasks) {
  const width = canvas.width;
  const height = canvas.height;

  // Create a new canvas for applying the background blur effect.
  bgCanvas.width = width;
  bgCanvas.height = height;

  // Copy the image data to the background canvas.
  bgCanvasCtx.putImageData(imageData, 0, 0);

  // Apply the blur effect to the background based on confidence masks.
  // You can adjust the level of blur by changing the parameters.
  // Here, we use a simple box blur as an example.

  const blurRadius = 10; // Adjust the blur radius as needed.
  const iterations = 3; // Adjust the number of iterations as needed.

  if (isSafari()) {
    bgCanvas.style.filter = `blur(${blurRadius}px)`;
    bgCanvas.style.webkitFilter = `blur(${blurRadius}px)`;
    bgCanvasCtx.drawImage(bgCanvas, 0, 0);
  } else {
    // Apply an alternative filter (e.g., CSS blur) for other browsers.
    for (let i = 0; i < iterations; i++) {
      bgCanvasCtx.filter = `blur(${blurRadius}px)`;
      bgCanvasCtx.drawImage(bgCanvas, 0, 0);
    }
    bgCanvasCtx.filter = "none";
  }
  // Return the background-blurred image data.
  return bgCanvasCtx.getImageData(0, 0, width, height);
}


function isSafari() {
  // Use user agent detection to identify Safari.
  const ua = navigator.userAgent.toLowerCase();
  return ua.indexOf('safari') !== -1 && ua.indexOf('chrome') === -1;
}
