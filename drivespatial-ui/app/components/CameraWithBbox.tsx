"use client";

import { useRef, useEffect, useState } from "react";
import Image from "next/image";

interface CameraWithBboxProps {
  imagePath: string;
  cameraName: string;
  bboxes?: number[][];
  labels?: string[];
  trackIds?: string[];
}

/**
 * Camera image with 2D bounding box overlays
 */
export default function CameraWithBbox({
  imagePath,
  cameraName,
  bboxes = [],
  labels = [],
  trackIds = [],
}: CameraWithBboxProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  /**
   * Generate consistent color from track ID (same as 3D viewer)
   */
  const getColorFromId = (trackId: string): string => {
    let hash = 0;
    for (let i = 0; i < trackId.length; i++) {
      hash = trackId.charCodeAt(i) + ((hash << 5) - hash);
    }
    // Generate bright, saturated colors
    const h = Math.abs(hash % 360);
    const s = 70 + (Math.abs(hash) % 30); // 70-100% saturation
    const l = 50 + (Math.abs(hash >> 8) % 20); // 50-70% lightness
    
    return `hsl(${h}, ${s}%, ${l}%)`;
  };

  useEffect(() => {
    if (!imgLoaded || !canvasRef.current || !imgRef.current || !bboxes || bboxes.length === 0) {
      return;
    }

    const canvas = canvasRef.current;
    const img = imgRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to match container size
    const container = canvas.parentElement;
    if (!container) return;
    
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scaling for object-contain behavior
    const containerAspect = canvas.width / canvas.height;
    const imageAspect = img.naturalWidth / img.naturalHeight;
    
    let displayWidth, displayHeight, offsetX, offsetY;
    
    if (containerAspect > imageAspect) {
      // Container is wider - image is limited by height
      displayHeight = canvas.height;
      displayWidth = displayHeight * imageAspect;
      offsetX = (canvas.width - displayWidth) / 2;
      offsetY = 0;
    } else {
      // Container is taller - image is limited by width
      displayWidth = canvas.width;
      displayHeight = displayWidth / imageAspect;
      offsetX = 0;
      offsetY = (canvas.height - displayHeight) / 2;
    }
    
    const scaleX = displayWidth / img.naturalWidth;
    const scaleY = displayHeight / img.naturalHeight;

    // Draw each bounding box
    bboxes.forEach((bbox, idx) => {
      if (!bbox || bbox.length < 4) return;

      const [xMin, yMin, xMax, yMax] = bbox;
      const label = labels[idx] || "unknown";
      const trackId = trackIds[idx] || "";

      // Scale coordinates to display size with letterbox offset
      const x1 = xMin * scaleX + offsetX;
      const y1 = yMin * scaleY + offsetY;
      const x2 = xMax * scaleX + offsetX;
      const y2 = yMax * scaleY + offsetY;
      const w = x2 - x1;
      const h = y2 - y1;

      // Get color based on track ID
      const labelStr = typeof label === 'string' ? label : String(label);
      const trackIdStr = typeof trackId === 'string' ? trackId : String(trackId || "");
      const color = trackIdStr ? getColorFromId(trackIdStr) : "#ffffff";

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      // Draw label: <class name> <5 chars of ID>
      const shortTrackId = trackIdStr.substring(0, 5);
      const labelText = `${shortTrackId}`;
      ctx.font = "14px Arial";
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const textHeight = 18;

      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight + 2);

      // Draw label text
      ctx.fillStyle = color;
      ctx.fillText(labelText, x1 + 4, y1 - 6);
    });
  }, [imgLoaded, bboxes, labels, trackIds]);

  // Redraw on container resize
  useEffect(() => {
    if (!containerRef.current) return;
    
    const resizeObserver = new ResizeObserver(() => {
      // Trigger redraw by setting imgLoaded state
      setImgLoaded(prev => prev);
    });
    
    resizeObserver.observe(containerRef.current);
    
    return () => resizeObserver.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="relative w-full h-full">
      {/* Image */}
      <Image
        ref={imgRef}
        src={`/api/hpc-image?path=${encodeURIComponent(imagePath)}`}
        alt={cameraName}
        fill
        className="object-contain"
        onLoad={() => setImgLoaded(true)}
        onError={() => console.error(`Failed to load image: ${cameraName}`)}
        unoptimized
      />

      {/* Canvas for bounding boxes */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 10 }}
      />

      {/* Camera label */}
      <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono">
        {cameraName}
      </div>
    </div>
  );
}
