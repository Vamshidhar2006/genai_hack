import React, { useState } from "react";
import UploadForm from "./UploadForm";
import PipelineStageLoader from "./PipelineStageLoader";
import AnalysisResult from "./AnalysisResult";

const API_BASE_URL = "http://localhost:8000";

function makeDemoAnnotatedImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.fillRect(img.width * 0.4, img.height * 0.35, img.width * 0.14, img.height * 0.12);
      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 16px sans-serif";
      ctx.fillText("BLURRED", img.width * 0.4, img.height * 0.34);

      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3;
      ctx.strokeRect(img.width * 0.15, img.height * 0.45, img.width * 0.25, img.height * 0.3);
      ctx.fillStyle = "#10b981";
      ctx.fillText("Car 0.94", img.width * 0.15, img.height * 0.44);

      resolve(canvas.toDataURL("image/jpeg", 0.9));
    };
    img.src = src;
  });
}

function Home({ useMockMode }) {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [userPrompt, setUserPrompt] = useState(
    "Describe traffic flow, road conditions, and pedestrian safety."
  );
  const [stage, setStage] = useState("idle");
  const [stage1Data, setStage1Data] = useState(null);
  const [stage2Data, setStage2Data] = useState(null);
  const [error, setError] = useState("");

  const handleFileSelect = (file) => {
    if (!file) return;
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setStage("idle");
    setStage1Data(null);
    setStage2Data(null);
    setError("");
  };

  const handleClear = () => {
    setImageFile(null);
    setPreviewUrl("");
    setStage("idle");
    setStage1Data(null);
    setStage2Data(null);
    setError("");
  };

  const getUploadBlob = async () => {
    if (imageFile instanceof Blob && imageFile.size > 0) {
      return imageFile;
    }
    if (previewUrl) {
      const resp = await fetch(previewUrl);
      const blob = await resp.blob();
      if (blob.size > 0) return blob;
    }
    throw new Error("Please select a valid street image before running Live API.");
  };

  const handleRunPipeline = async () => {
    setError("");
    setStage1Data(null);
    setStage2Data(null);

    try {
      setStage("yolo_blur_loading");
      let s1;
      let imageForVlm = "";

      if (useMockMode) {
        await new Promise((r) => setTimeout(r, 1800));
        const demoImg = await makeDemoAnnotatedImage(previewUrl);
        imageForVlm = demoImg;
        s1 = {
          session_id: "demo_01",
          blurred_detected_image: demoImg,
          faces_blurred_count: 2,
          plates_blurred_count: 1,
          detections: [
            { label: "Car", confidence: 0.94, count: 3 },
            { label: "Person", confidence: 0.88, count: 2 },
          ],
        };
      } else {
        const uploadBlob = await getUploadBlob();
        const formData = new FormData();
        formData.append("file", uploadBlob, imageFile?.name || "street.jpg");
        formData.append("enable_blur", "true");
        formData.append("street_only", "false");

        const res1 = await fetch(`${API_BASE_URL}/api/preprocess-detect`, {
          method: "POST",
          body: formData,
        });

        if (!res1.ok) {
          let detail = "Stage 1 failed in Python backend";
          try {
            const errJson = await res1.json();
            if (errJson?.detail) detail = errJson.detail;
          } catch {
            // Keep default message if response is not JSON
          }
          throw new Error(detail);
        }

        const rawS1 = await res1.json();
        const annotatedOrBlurred = rawS1.annotated_image || rawS1.blurred_image || "";
        imageForVlm = rawS1.blurred_image || rawS1.annotated_image || "";

        s1 = {
          ...rawS1,
          blurred_detected_image: annotatedOrBlurred,
          faces_blurred_count: rawS1.blurred_count ?? 0,
        };
      }

      setStage1Data(s1);
      setStage("vlm_loading");

      let s2;
      if (useMockMode) {
        await new Promise((r) => setTimeout(r, 1800));
        s2 = {
          description: `Scene analysis for "${userPrompt}": The street shows moderate vehicle movement with detected cars and pedestrians. Faces and license plates were blurred before analysis.`,
          pdf_download_url: "#demo-pdf",
        };
      } else {
        const res2 = await fetch(`${API_BASE_URL}/api/analyze-vlm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: imageForVlm,
            counts: s1.counts || {},
            detections: s1.detections || [],
            focus_prompt: userPrompt,
          }),
        });

        if (!res2.ok) {
          let detail = "Stage 2 failed in Python backend";
          try {
            const errJson = await res2.json();
            if (errJson?.detail) detail = errJson.detail;
          } catch {
            // Keep default message if response is not JSON
          }
          throw new Error(detail);
        }

        const rawS2 = await res2.json();
        s2 = {
          ...rawS2,
          description: rawS2.report || rawS2.description || "",
        };
      }

      setStage2Data(s2);
      setStage("complete");
    } catch (err) {
      setError(err.message);
      setStage("idle");
    }
  };

  const isRunning = stage === "yolo_blur_loading" || stage === "vlm_loading";

  return (
    <div className="home-layout">
      <UploadForm
        imageFile={imageFile}
        previewUrl={previewUrl}
        userPrompt={userPrompt}
        setUserPrompt={setUserPrompt}
        onFileSelect={handleFileSelect}
        onClear={handleClear}
        onRunPipeline={handleRunPipeline}
        disabled={isRunning}
      />

      {error && <div className="error-box">{error}</div>}

      <PipelineStageLoader stage={stage} previewUrl={previewUrl} />

      <AnalysisResult
        stage1Data={stage1Data}
        stage2Data={stage2Data}
        userPrompt={userPrompt}
      />
    </div>
  );
}

export default Home;